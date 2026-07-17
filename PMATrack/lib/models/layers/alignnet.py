
from __future__ import annotations
from typing import Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from einops import rearrange
def _make_proj(dim):
    layers = nn.Sequential(
        nn.Linear(dim, dim),
        nn.ReLU(inplace=True),
        nn.Linear(dim, dim)
    )

    # ====== 初始化 ======
    for m in layers:
        if isinstance(m, nn.Linear):
            # Kaiming 更适合 ReLU 激活
            init.kaiming_uniform_(m.weight, a=0, nonlinearity="relu")
            if m.bias is not None:
                    init.zeros_(m.bias)
    return layers

class TargetExpert(nn.Module):
    """response-based coarse shift estimator (OT-enhanced) -> 输出 coarse_dxdy, response maps, scale_hint"""
    def __init__(self, dim, beta=10.0, activate="softmax", temperature=0.05, margin_ratio=1.0,
                 sinkhorn_eps=0.02, sinkhorn_iters=50, sinkhorn_stabilize=True):
        super().__init__()
        self.beta = beta
        self.temperature = temperature

        self.rgb_proj = _make_proj(dim)
        self.tir_proj = _make_proj(dim)
        self.norm1 = nn.LayerNorm(dim)

        if activate == "sigmoid":
            self.activate = torch.sigmoid
        elif activate == "softmax":
            self.activate = lambda x: F.softmax(x, dim=-1)
        else:
            self.activate = lambda x: x

        # Corr Refiner（维持你的结构与输出维度）
        self.refiner = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 5)
        )

        # ranges（保留）
        self.x_range = 1 + margin_ratio
        self.y_range = 1 + margin_ratio

        # buffers
        self.register_buffer("grid_x", None, persistent=False)
        self.register_buffer("grid_y", None, persistent=False)

        # OT params
        self.sinkhorn_eps = sinkhorn_eps          # ε: 熵正则强度（越大越平滑）
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_stabilize = sinkhorn_stabilize

    @torch.no_grad()
    def _prepare_coords(self, N, device):
        """生成 H×W 的网格坐标；需要时创建/缓存。"""
        H = W = int(N ** 0.5)
        if (self.grid_x is None) or (self.grid_x.numel() != N):
            gy, gx = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing='ij'
            )
            self.grid_y = gy.reshape(-1).float()  # [N]
            self.grid_x = gx.reshape(-1).float()  # [N]
        return H, W, self.grid_y, self.grid_x

    def _sinkhorn(self, a, b, C, eps, iters):
        """
        a,b: [B,N] 概率分布（和为1）
        C:   [N,N] 代价矩阵（共享，欧式距离平方）
        返回运输矩阵 P: [B,N,N]
        """
        B, N = a.shape
        # K = exp(-C/eps) ，可选 log-domain 稳定
        if self.sinkhorn_stabilize:
            # 直接在实数域迭代也可行（N~几百量级），数值更直观
            K = torch.exp(-C / eps)  # [N,N]
            # 初始化
            u = torch.ones(B, N, device=a.device) / N
            v = torch.ones(B, N, device=a.device) / N
            Kb = K  # 共享
            KT = K.transpose(0,1)

            for _ in range(iters):
                Kv = torch.matmul(v, Kb.T)         # [B,N]
                u = a / (Kv + 1e-12)
                Ku = torch.matmul(u, Kb)           # [B,N]
                v = b / (Ku + 1e-12)

            # P = diag(u) K diag(v)
            P = u.unsqueeze(-1) * Kb.unsqueeze(0) * v.unsqueeze(-2)  # [B,N,N]
            return P
        else:
            # 简单稳定版本：与上面类似
            K = torch.exp(-C / eps)
            u = torch.ones(B, N, device=a.device) / N
            v = torch.ones(B, N, device=a.device) / N
            for _ in range(iters):
                u = a / (torch.matmul(v, K.T) + 1e-12)
                v = b / (torch.matmul(u, K) + 1e-12)
            P = u.unsqueeze(-1) * K.unsqueeze(0) * v.unsqueeze(-2)
            return P

    def _ot_to_shift_corr(self, P, H, W, gy, gx):
        """
        将运输矩阵 P[B,N,N] 聚合成位移直方图 corr[B, 2H-1, 2W-1]
        其中每个 (i->j) 的质量累加到 (dy=g_y[j]-g_y[i], dx=g_x[j]-g_x[i]) 处
        """
        B, N, _ = P.shape
        device = P.device

        # [N] -> [N,1] & [1,N] 便于广播
        y_i = gy.view(N, 1)
        y_j = gy.view(1, N)
        x_i = gx.view(N, 1)
        x_j = gx.view(1, N)

        # [N,N] 整数位移
        dy = (y_j - y_i).to(torch.int64)  # [N,N]
        dx = (x_j - x_i).to(torch.int64)  # [N,N]

        # 平移到正索引
        oy = H - 1
        ox = W - 1
        iy = (dy + oy).view(-1)  # [N*N]
        ix = (dx + ox).view(-1)  # [N*N]
        flat_idx = iy * (2 * W - 1) + ix  # [N*N]

        # 聚合
        corr = torch.zeros(B, (2 * H - 1) * (2 * W - 1), device=device)
        P_flat = P.view(B, -1)  # [B, N*N]
        corr.scatter_add_(dim=1, index=flat_idx.unsqueeze(0).expand(B, -1), src=P_flat)
        corr = corr.view(B, 2 * H - 1, 2 * W - 1)
        return corr

    def forward(self, rgb_z, rgb_x, tir_z, tir_x):
        # 1) feature proj
        rgb_z_p, rgb_x_p = self.rgb_proj(self.norm1(rgb_z)), self.rgb_proj(self.norm1(rgb_x))
        tir_z_p, tir_x_p = self.tir_proj(self.norm1(tir_z)), self.tir_proj(self.norm1(tir_x))

        # 2) similarity
        rgb_target = rgb_z_p @ rgb_x_p.transpose(1, 2)  # [B,N,N]
        tir_target = tir_z_p @ tir_x_p.transpose(1, 2)  # [B,N,N]

        # 3) activation -> produce per-spatial map (aggregate over template dim)
        rgb_act = torch.softmax(rgb_target, dim=-1).max(dim=1).values  # [B, N]
        tir_act = torch.softmax(tir_target, dim=-1).max(dim=1).values  # [B, N]

        B, N = rgb_act.shape
        device = rgb_act.device
        H, W, gy, gx = self._prepare_coords(N, device)

        # --- OT: a,b 归一化为概率分布 ---
        # 防止全零：加一个极小常数再 normalize
        a = rgb_act + 1e-8
        b = tir_act + 1e-8
        a = a / a.sum(dim=-1, keepdim=True)  # [B,N]
        b = b / b.sum(dim=-1, keepdim=True)  # [B,N]

        # 代价矩阵 C[N,N] = ||p_i - p_j||^2
        # 用 (x,y) 欧氏距离平方
        # (x^2+y^2) - 2(x_i x_j + y_i y_j) + 常数，可直接用展开式
        gx2 = gx**2
        gy2 = gy**2
        # ||p_i||^2 + ||p_j||^2 - 2 <p_i, p_j>
        C = (gx2.view(-1,1) + gy2.view(-1,1)) + (gx2.view(1,-1) + gy2.view(1,-1)) \
            - 2*(gx.view(-1,1)*gx.view(1,-1) + gy.view(-1,1)*gy.view(1,-1))
        C = C.float()  # [N,N]

        # Sinkhorn 得到运输矩阵 P[B,N,N]
        eps = self.sinkhorn_eps if self.sinkhorn_eps is not None else 0.02
        iters = self.sinkhorn_iters
        P = self._sinkhorn(a, b, C, eps, iters)  # [B,N,N]

        # 将 P 聚合为位移直方图（与原先 corr 大小一致）
        corr_ot = self._ot_to_shift_corr(P, H, W, gy, gx)  # [B, 2H-1, 2W-1]

        # ---- refiner 输入（保持你的 refiner 维度假设）----
        offset = self.refiner(corr_ot.unsqueeze(1))  # [B,5]

        # split and post-process（保持原逻辑）
        dx_px = offset[:, 0]
        dy_px = offset[:, 1]
        dlog_sx = offset[:, 2]
        dlog_sy = offset[:, 3]
        raw_conf = offset[:, 4:5]
        conf = torch.sigmoid(raw_conf).squeeze(-1)  # [B]

        dx_norm = torch.tanh(dx_px)
        dy_norm = torch.tanh(dy_px)

        max_log = math.log(4)  # ln(4)
        dlog_sx = torch.tanh(dlog_sx) * max_log
        dlog_sy = torch.tanh(dlog_sy) * max_log

        return torch.stack([dx_norm, dy_norm, dlog_sx, dlog_sy], dim=-1), rgb_target, tir_target, conf


class FrequencyDecomposer(nn.Module):
    def __init__(self, k_low: int = 5, k_high: int = 3):
        super().__init__()
        self.k_low = k_low
        self.k_high = k_high

    @staticmethod
    def _to_nchw(x):  # x: B,H,W,D or B,HW,D
        if x.dim() == 4:
            B,H,W,D = x.shape
            return x.permute(0,3,1,2).contiguous(), (B,H,W,D)
        elif x.dim() == 3:
            B,HW,D = x.shape
            H = W = int(HW ** 0.5)
            assert H*W == HW, "HW 不是正方形，请先提供(H,W)或改写这里的reshape逻辑。"
            x4 = x.view(B,H,W,D).permute(0,3,1,2).contiguous()
            return x4, (B,H,W,D)
        else:
            raise ValueError("feat must be [B,H,W,D] or [B,HW,D]")

    @staticmethod
    def _to_bhwd(x, shape):
        B,H,W,D = shape
        return x.permute(0,2,3,1).contiguous().view(B,H*W,D)

    def forward(self, x):
        x4, shape = self._to_nchw(x)  # [B,D,H,W]
        low  = F.avg_pool2d(x4, kernel_size=self.k_low,  stride=1, padding=self.k_low//2)
        base = F.avg_pool2d(x4, kernel_size=self.k_high, stride=1, padding=self.k_high//2)
        high = x4 - base
        low  = self._to_bhwd(low,  shape)
        high = self._to_bhwd(high, shape)
        return low, high


class FreqGate(nn.Module):
    def __init__(self, dim, hidden=128, prefer_low_for_tir=True):
        super().__init__()
        self.prefer_low_for_tir = prefer_low_for_tir
        self.mlp = nn.Sequential(
            nn.Linear(2*dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2)
        )
        with torch.no_grad():
            last = self.mlp[-1]
            last.weight.zero_()
            bias = torch.tensor([0.5, -0.5]) if prefer_low_for_tir else torch.tensor([-0.5, 0.5])
            last.bias.copy_(bias)

    def forward(self, rgb_feat_bhwd, tir_feat_bhwd):
        rgb_gap = rgb_feat_bhwd.mean(dim=1)
        tir_gap = tir_feat_bhwd.mean(dim=1)
        x = torch.cat([rgb_gap, tir_gap], dim=-1)
        logits = self.mlp(x)                 # [B,2]
        band_w = F.softmax(logits, dim=-1)   # [B,2]
        return band_w


class PyramidCorrHead(nn.Module):
    """
    金字塔相关图 -> 多尺度一致性 -> 偏移预测
    输入: corr [B, HW, HW], 假设 HW=H*W 且 H=W
    输出: [B,5] (dx, dy, dlog_sx, dlog_sy, conf)
    """
    def __init__(self, HW, base_channels=64, num_scales=3):
        super().__init__()
        H = W = HW
        self.H, self.W = H, W
        self.num_scales = num_scales

        # 每个尺度用同构卷积块
        self.blocks = nn.ModuleList()
        for i in range(num_scales):
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(1, base_channels, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                )
            )
        # 融合层：concat后降维
        self.fuse = nn.Sequential(
            nn.Conv2d(base_channels*num_scales, base_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        # GAP + 线性预测
        self.head = nn.Linear(base_channels, 5)

    def forward(self, corr):
        B, HW, HW2 = corr.shape
        assert HW == HW2
        x = corr.view(B,1,self.H,self.W)   # [B,1,H,W]

        feats = []
        cur = x
        for i, blk in enumerate(self.blocks):
            f = blk(cur)                 # [B,C,h,w]
            if f.shape[2:] != (self.H,self.W):
                f_up = F.interpolate(f, size=(self.H,self.W), mode="bilinear", align_corners=False)
            else:
                f_up = f
            feats.append(f_up)
            # 下采样供下一层
            cur = F.avg_pool2d(cur, kernel_size=2, stride=2)

        feat_cat = torch.cat(feats, dim=1)   # [B,C*num_scales,H,W]
        fused = self.fuse(feat_cat)          # [B,C,H,W]
        pooled = fused.mean(dim=(2,3))       # GAP [B,C]
        out = self.head(pooled)              # [B,5]
        return out
# 如果你的工程里已有 _make_proj，这里会被覆盖；没有的话就走这个最简实现
def _make_proj2(dim: int):
    return nn.Linear(dim, dim, bias=False)

class StructExpert(nn.Module):
    def __init__(self, dim, HW, alpha_init=0.5, hidden_dim=128, k_low=5, k_high=3):
        super().__init__()
        self.rgb_proj = _make_proj2(dim=dim)
        self.tir_proj = _make_proj2(dim=dim)
        self.alpha1 = nn.Parameter(torch.tensor(alpha_init))
        self.alpha2 = nn.Parameter(torch.tensor(alpha_init))
        self.HW = HW
        self.dim = dim

        # 频率模块
        self.freq = FrequencyDecomposer(k_low=k_low, k_high=k_high)
        self.band_gate_rgb = FreqGate(dim, hidden=hidden_dim, prefer_low_for_tir=False) # RGB倾向高频
        self.band_gate_tir = FreqGate(dim, hidden=hidden_dim, prefer_low_for_tir=True)  # TIR倾向低频

        self.norm1 = nn.LayerNorm(dim)
        
        self.conv_head = PyramidCorrHead(HW, base_channels=64, num_scales=3)

    @staticmethod
    def _to_bhwd(x, HW, D):
        if x.dim() == 4:   # [B,H,W,D]
            B,H,W,Din = x.shape
            assert Din == D
            return x.view(B, H*W, D)
        elif x.dim() == 3: # [B,HW,D]
            B,HW_in,Din = x.shape
            assert HW_in == HW and Din == D
            return x
        else:
            raise ValueError("feat must be [B,H,W,D] or [B,HW,D]")

    def forward(self, feat1, feat2):
        B = feat1.shape[0]
        D = self.dim
        HW = self.HW

        f1 = self._to_bhwd(feat1, HW, D)  # [B,HW,D]
        f2 = self._to_bhwd(feat2, HW, D)

        f1 = self.rgb_proj(f1)
        f2 = self.tir_proj(f2)

        rgb_low,  rgb_high  = self.freq(f1)   # [B,HW,D]
        tir_low,  tir_high  = self.freq(f2)

        rgb_band_w = self.band_gate_rgb(rgb_low,  rgb_high)   # [B,2]
        tir_band_w = self.band_gate_tir(tir_low,  tir_high)   # [B,2]

        def corr_softmax(a, b):
            a = self.norm1(a)
            b = self.norm1(b)
            c = torch.bmm(a, b.transpose(1, 2))  # [B,HW,HW]
            return F.softmax(c, dim=-1)

        corr_low  = corr_softmax(rgb_low,  tir_low)   # [B,HW,HW]
        corr_high = corr_softmax(rgb_high, tir_high)  # [B,HW,HW]

        w_low  = torch.sqrt(rgb_band_w[:, 0:1] * tir_band_w[:, 0:1])  # [B,1]
        w_high = torch.sqrt(rgb_band_w[:, 1:2] * tir_band_w[:, 1:2])  # [B,1]
        w_low_map  = w_low.view(B, 1, 1)
        w_high_map = w_high.view(B, 1, 1)

        corr = w_low_map * corr_low + w_high_map * corr_high  # [B,HW,HW]

        offset_scale = self.conv_head(corr)  # [B,5]

        dx_px   = offset_scale[:, 0]
        dy_px   = offset_scale[:, 1]
        dlog_sx = offset_scale[:, 2]
        dlog_sy = offset_scale[:, 3]
        raw_conf = offset_scale[:, 4:5]

        scale_conf = torch.sigmoid(raw_conf).squeeze(-1)  # [B]
        dx_norm = torch.tanh(dx_px)
        dy_norm = torch.tanh(dy_px)
        max_log = math.log(4)
        dlog_sx = torch.tanh(dlog_sx) * max_log
        dlog_sy = torch.tanh(dlog_sy) * max_log

        return torch.stack([dx_norm, dy_norm, dlog_sx, dlog_sy], dim=-1), scale_conf, corr


def _autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class _Conv(nn.Module):
    """Conv-BN-Act"""
    default_act = nn.GELU()
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, _autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn   = nn.BatchNorm2d(c2, eps=1e-3, momentum=0.03)
        self.act  = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
    def forward(self, x): return self.act(self.bn(self.conv(x)))

class _DWConv(_Conv):
    """Depthwise Conv"""
    def __init__(self, c1, c2, k=3, s=1, d=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)

class CMRF(nn.Module):
    """
    Cascade Multi-Receptive Fields
    - 先PW降通道，再对偶数通道做级联DWConv获取多感受野，
      与奇数通道逐元素相加后拼接，最后PW融合。
    - 与 TinyU-Net 一致的轻量策略。
    Args:
        c1: in_channels
        c2: out_channels
        N:  cascade个数（DWConv块数=N-1）
        e:  中间通道压缩比例（相对c2）
        shortcut: 是否与输入残差相加（c1==c2时启用）
    """
    def __init__(self, c1, c2, N=8, e=0.5, shortcut=True):
        super().__init__()
        assert N >= 2, "N should be >= 2"
        self.N   = N
        self.add = (shortcut and c1 == c2)

        mid = c2 // N                 # 先收缩到 c2/N 通道
        self.pw1 = _Conv(c1, mid, k=1, s=1)
        # 级联链路里每个节点的通道数保持一致
        self.dw_chain = nn.ModuleList([_DWConv(mid // 2, mid // 2, k=3, s=1, act=False) for _ in range(N-1)])
        # 融合后通道恢复到 c2
        self.pw2 = _Conv((N) * (mid // 2), c2, k=1, s=1)

    def forward(self, x):
        x_res = x
        x = self.pw1(x)                # [B, mid, H, W]
        # 奇偶拆分（与 TinyU-Net 写法一致：偶/奇交替）
        x_even = x[:, 1::2, :, :]       # [B, mid//2, H, W]
        x_odd  = x[:, 0::2, :, :]       # [B, mid//2, H, W]

        # 级联：不断在上一层输出上做DWConv（获取更大/不同感受野）
        feats = [x_odd + x_even]        # 线性混合（mixup风格）
        cur   = x_even
        for dw in self.dw_chain:
            cur = dw(cur)
            feats.append(cur)

        y = torch.cat(feats, dim=1)     # [B, (N)*(mid//2), H, W]
        y = self.pw2(y)                 # [B, c2, H, W]
        return (x_res + y) if self.add else y
    

# ---------- TinyU-Net 风格主干（多尺度 + 跳连） ----------
class CMRF_TinyUNet(nn.Module):
    """
    一个很轻的 U 形：Enc: CMRF+下采样 × L 层；Dec: 上采样+拼接+CMRF × L 层
    仅在二维整图上做卷积，不用相关图/注意力。
    """
    def __init__(self, in_ch, out_ch, base_ch, levels=3, cmrf_N=8):
        super().__init__()
        assert levels in [2,3,4]  # 够用且稳
        chs = [base_ch*(2**i) for i in range(levels)]  # e.g., 64,128,256

        # 编码
        self.enc_blocks = nn.ModuleList()
        self.pools      = nn.ModuleList()
        c_in = in_ch
        for c in chs:
            self.enc_blocks.append(CMRF(c_in, c, N=cmrf_N, shortcut=False))
            self.pools.append(nn.MaxPool2d(2,2))
            c_in = c

        # 底部
        self.bottleneck = CMRF(chs[-1], chs[-1], N=cmrf_N, shortcut=True)

        # 解码
        self.up_blocks = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in reversed(range(levels)):
            c_skip = chs[i]
            c_up_in = chs[i] if i==levels-1 else chs[i+1]  # 来自上一层解码输出
            self.up_blocks.append(nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False))
            self.dec_blocks.append(CMRF(c_up_in + c_skip, chs[i], N=cmrf_N, shortcut=False))

        # 输出映射
        self.head = _Conv(chs[0], out_ch, k=1, s=1, act=False)

    @staticmethod
    def _match(hw_ref, x):
        """双线性调整到参考尺寸（处理偶/奇尺寸不同步）"""
        return F.interpolate(x, size=hw_ref, mode='bilinear', align_corners=False)

    def forward(self, x):
        skips = []
        h, w = x.shape[-2:]
        # 编码
        for enc, pool in zip(self.enc_blocks, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        # 底部
        x = self.bottleneck(x)

        # 解码（与 skip 拼接）
        for up, dec, skip in zip(self.up_blocks, self.dec_blocks, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = self._match(skip.shape[-2:], x)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        # 输出
        y = self.head(x)
        if y.shape[-2:] != (h, w):
            y = self._match((h, w), y)
        return y

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
    
class MHSA2D(nn.Module):
    """简单的 MHSA，用在 [B, HW, D] 上"""
    def __init__(self, dim, num_heads=8, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # x: [B, HW, D]
        B, N, C = x.shape
        qkv = self.qkv(x)  # [B, N, 3C]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]   # 各: [B, heads, N, head_dim]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  # [B, heads, N, head_dim]
        out = out.transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class DiffTransformerBlock(nn.Module):
    """
    标准 Transformer Block，但专门用在“跨模态差分 token”上。
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MHSA2D(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)

        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        # x: [B, HW, D]
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    
# ---------- DetailExpert：使用 TinyU-Net 风格主干 ----------
class DetailExpert(nn.Module):
    """
    使用 TinyU-Net 风格主干的细化器
    """
    def __init__(self, dim, hidden_channels=256, k_corr=64,
                 use_corr_enhance=True, use_resp_enhance=True,
                 cmrf_N=8, levels=4, base_ratio=1.0):
        super().__init__()
        self.k_corr = k_corr
        self.use_resp_enhance = use_resp_enhance
        self.use_corr_enhance = use_corr_enhance

        in_ch   = dim
        out_ch  = dim
        base_ch = max(32, int(dim*base_ratio))

        self.backbone = CMRF_TinyUNet(in_ch=in_ch, out_ch=out_ch,
                                      base_ch=base_ch, levels=levels, cmrf_N=cmrf_N)
        # 3) Global 分支：多层差分 Transformer
        self.global_blocks = nn.ModuleList([
            DiffTransformerBlock(
                dim=dim*2,
                num_heads=8,
                mlp_ratio=4.,
                drop=0.,
                attn_drop=0.
            )
            for _ in range(1)
        ])
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2*dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 5)
        )

    def forward(self, feat1, feat2, resp1=None, resp2=None, corr=None, coarse_delta=None, scale_hint=None, scale_pred=None):
        B, HW, D = feat1.shape
        side = int(HW ** 0.5)
        assert side * side == HW, f"DetailExpert expects square feature map, but got HW={HW}"

        # x = torch.cat([feat1, feat2], dim=-1).view(B, D, side, side)
        feat1 = feat1.view(B, D, side, side)
        feat2 = feat2.view(B, D, side, side)
        feat1 = self.backbone(feat1)
        feat2 = self.backbone(feat2)
        x = torch.cat([feat1, feat2], dim=1).view(B, side*side, 2*D)

        for blk in self.global_blocks:
            x = blk(x)   
        x = x.view(B, 2*D, side, side)    
        post = self.regressor(x)
        dx_res, dy_res, dlog_sx_res, dlog_sy_res, conf = post.unbind(dim=-1)

        dx_norm = torch.tanh(dx_res)
        dy_norm = torch.tanh(dy_res)
        max_log = math.log(4)
        dlog_sx_res = torch.tanh(dlog_sx_res) * max_log
        dlog_sy_res = torch.tanh(dlog_sy_res) * max_log

        residuals  = torch.stack([dx_norm, dy_norm, dlog_sx_res, dlog_sy_res], dim=-1)
        final_conf = torch.sigmoid(conf)
        return residuals, final_conf

# ---------- 工具函数 ----------
def count_trainable_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

# ---------- 门控网络 ----------
class RoutingGate(nn.Module):
    def __init__(self, d_in, hidden=128, use_gumbel=True, tau=1.0, hard_routing=True):
        super().__init__()
        self.use_gumbel = use_gumbel
        self.tau = tau
        self.hard_routing = hard_routing
        self.mlp = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3)  # logits for shift / scale / warp
        )
    def forward(self, feat1, feat2):
        stats = torch.cat([feat1[:,:,0], feat2[:,:,0]],1)
        logits = self.mlp(stats)     
        logits = F.gumbel_softmax(logits, tau=self.tau, hard=self.hard_routing)# [B, 3]
        return logits

class NeuroNet(nn.Module):
    def __init__(self,
                 height,
                 width,
                 embed_dim,
                 gate_dim=512,
                 gate_hidden=128,
                 gate_tau=1.0,
                 gate_hard=False,
                 lambda_param_cost=0.01,   # 抽象惩罚系数 λ
                 ):
        super().__init__()
        actual_width = height * width
        self.shift_net = TargetExpert(dim=embed_dim, activate="softmax")
        self.scale_net = StructExpert(dim=embed_dim, HW=actual_width, alpha_init=0.5, hidden_dim=actual_width*4)
        self.warp_net  = DetailExpert(dim=embed_dim, cmrf_N=8, levels=4, base_ratio=256/768)

        self._gate = RoutingGate(d_in=gate_dim, hidden=gate_hidden, use_gumbel=True, tau=gate_tau, hard_routing=gate_hard)
        
        self.lambda_param_cost = lambda_param_cost
        self.eps = 1e-8
        self.cost_w = 1.0
        self.scale_w = 1.0
        self.lamba_entropy = 1e-3
        self.set_branch_costs()
        # TargetExpert 1.529 GFlops 
        # StructExpert 8.896 GFlops 
        # DetailExpert 25.536 GFlops 
    @torch.no_grad()
    def set_branch_costs(self,
                        mode: str = "max",
                        gflops=None):
        # 1) 读取 GFLOPs（默认常量或用户传入）
        if gflops is None:
            c_shift = 1.529
            c_scale = 8.896
            c_warp  = 15.110
        costs = torch.tensor([c_shift, c_scale, c_warp], dtype=torch.float32)
        # 2) 归一化
        if mode == "max":
            denom = costs.max().clamp_min(1e-8)
            costs = costs / denom
        elif mode == "sum":
            denom = costs.sum().clamp_min(1e-8)
            costs = costs / denom
            
        self.register_buffer("branch_costs", costs)  # [3]

        
    def forward(self, rgb_z, rgb_x, tir_z, tir_x, gt=None, return_all=False, layer_index=0, previous_preds=None):

        gate_logits = self._gate(rgb_x, tir_x)
        # Shift
        shift_delta, rgb_target, tir_target, _ = self.shift_net(rgb_z, rgb_x, tir_z, tir_x)
        # Scale
        scale_vec, _, _ = self.scale_net(rgb_x, tir_x)  # [B,4], [B], [B,HW,HW]
        # Warp（残差）
        warp_residual, _ = self.warp_net(rgb_x, tir_x)  

        #   Expert 0 (Shift): 仅平移（尺度设为0增量）
        dx_s, dy_s, lxs_s, lys_s  = shift_delta[:, 0], shift_delta[:, 1], shift_delta[:, 2], shift_delta[:, 3]
        #   Expert 1 (Scale): 平移 + 尺度（直接来自 scale 分支）
        dx_c, dy_c, lxs_c, lys_c = scale_vec[:, 0], scale_vec[:, 1], scale_vec[:, 2], scale_vec[:, 3]
        #   Expert 2 (Warp): 对 Shift+Scale 再做残差 refine
        dx_w, dy_w, lxs_w, lys_w = warp_residual[:, 0], warp_residual[:, 1], warp_residual[:, 2], warp_residual[:, 3]
        
        E0 = torch.stack([dx_s, dy_s, lxs_s, lys_s], dim=-1)  # [B,4]
        E1 = torch.stack([dx_c, dy_c, lxs_c,  lys_c], dim=-1)  # [B,4]
        E2 = torch.stack([dx_w, dy_w, lxs_w, lys_w], dim=-1)  # [B,4]
        
        experts = torch.stack([E0, E1, E2], dim=1)  # [B, 3, 4]
        pred = (gate_logits.unsqueeze(-1) * experts).sum(dim=1)  # [B,4]
        if layer_index == 0:
            pred = torch.stack([pred[:,0], 
                                pred[:,1], 
                                torch.zeros_like(pred[:,2]), 
                                torch.zeros_like(pred[:,3])], dim=-1)  # [B,4]
        elif layer_index==1:
            pred = pred + previous_preds[-1]
            experts = torch.stack([
                E0 + previous_preds[-1], 
                E1 + previous_preds[-1], 
                E2 + previous_preds[-1]
            ], dim=1)  # [B,3,4]
            
        else:
            pred = pred + previous_preds[-1]
            experts = torch.stack([
                E0 + previous_preds[-1], 
                E1 + previous_preds[-1], 
                E2 + previous_preds[-1]
            ], dim=1)  # [B,3,4]
        losses = {}
        aux = {
            "gate_logits": gate_logits,
            "experts": experts,
            "rgb_target": rgb_target, 
            "tir_target": tir_target,
        }

        # 若未提供 GT，则只做前向组合
        if gt is None:
            return (pred, losses, aux) if return_all else pred
        
        # 假设：gate 为概率 [B,3]；branch_costs 已在 set_branch_costs 中线性归一（/max）
        err0 = F.smooth_l1_loss(E0, gt, reduction='none')  # [B,4]
        err1 = F.smooth_l1_loss(E1, gt, reduction='none')
        err2 = F.smooth_l1_loss(E2, gt, reduction='none')

        def reduce_err(e):
            trans = e[:, :2].sum(dim=-1)     # dx, dy
            scale = e[:, 2:].sum(dim=-1)     # log sx, log sy
            return trans + self.scale_w * scale   # [B]

        l0, l1, l2 = reduce_err(err0), reduce_err(err1), reduce_err(err2)  # [B]

        # 前向与训练用同一 gate
        pred = (gate_logits.unsqueeze(-1) * experts).sum(dim=1)

        # 期望任务损失
        L_route = (gate_logits[:,0]*l0 + gate_logits[:,1]*l1 + gate_logits[:,2]*l2).mean()

        # 复杂度正则（线性、可解释）
        L_cost = self.lambda_param_cost * (gate_logits * self.branch_costs.unsqueeze(0)).sum(dim=-1).mean()

        # # 熵正则（鼓励硬路由则取正）
        # entropy = -(gate_logits.clamp_min(self.eps).log() * gate_logits).sum(dim=-1).mean()
        # L_ent = self.lamba_entropy * entropy

        L_total = L_route + L_cost 

        return (pred, L_total, aux) if return_all else (pred, L_total)
    
    @torch.no_grad()
    def infer(self,
            rgb_z, rgb_x, tir_z, tir_x,
            layer_index: int = 0,
            previous_preds=None,
            return_aux: bool = False,
            force_select: bool = True,    # 新增：是否强制
            selected_id: int = 0             # 新增：指定专家 0/1/2
            ):
        """
        硬路由推理：只激活被 gate 选中的专家并计算其输出。
        如果 force_select=True，则忽略 gate，所有样本走 selected_id 指定的专家。

        返回:
            pred: [B,4]
            aux(可选): dict
        """
        device = rgb_x.device
        B = rgb_x.shape[0]

        # ------------------------------------------------------------
        # 1) Gate 分支：可选跳过
        # ------------------------------------------------------------
        gate_logits = self._gate(rgb_x, tir_x)     # [B,3]

        if force_select:
            # 所有样本都走 selected_id 分支
            chosen = torch.full((B,), selected_id, device=device, dtype=torch.long)
        else:
            # 正常硬路由
            chosen = torch.argmax(gate_logits, dim=-1)

        # ------------------------------------------------------------
        # 2) 输出容器
        # ------------------------------------------------------------
        pred = torch.zeros(B, 4, device=device)

        # 上一层残差
        prev = previous_preds[-1] if (previous_preds is not None and len(previous_preds) > 0) else None

        # ------------------------------------------------------------
        # expert 0: shift-only
        # ------------------------------------------------------------
        idx0 = (chosen == 0).nonzero(as_tuple=True)[0]
        if idx0.numel() > 0:
            sd0, _, _, _ = self.shift_net(rgb_z[idx0], rgb_x[idx0], tir_z[idx0], tir_x[idx0])
            E0 = torch.stack([sd0[:,0], sd0[:,1], sd0[:,2], sd0[:,3]], dim=-1)

            if layer_index == 0:
                E0 = torch.stack([E0[:,0], E0[:,1],
                                torch.zeros_like(E0[:,2]),
                                torch.zeros_like(E0[:,3])], dim=-1)
            else:
                if prev is not None:
                    E0 = E0 + prev[idx0]

            pred[idx0] = E0

        # ------------------------------------------------------------
        # expert 1: scale
        # ------------------------------------------------------------
        idx1 = (chosen == 1).nonzero(as_tuple=True)[0]
        if idx1.numel() > 0:
            sv1, _, _ = self.scale_net(rgb_x[idx1], tir_x[idx1])
            E1 = torch.stack([sv1[:,0], sv1[:,1], sv1[:,2], sv1[:,3]], dim=-1)

            if layer_index == 0:
                E1 = torch.stack([E1[:,0], E1[:,1],
                                torch.zeros_like(E1[:,2]),
                                torch.zeros_like(E1[:,3])], dim=-1)
            else:
                if prev is not None:
                    E1 = E1 + prev[idx1]

            pred[idx1] = E1

        # ------------------------------------------------------------
        # expert 2: warp residual
        # ------------------------------------------------------------
        idx2 = (chosen == 2).nonzero(as_tuple=True)[0]
        if idx2.numel() > 0:
            wr2, _ = self.warp_net(rgb_x[idx2], tir_x[idx2])
            E2 = torch.stack([wr2[:,0], wr2[:,1], wr2[:,2], wr2[:,3]], dim=-1)

            if layer_index == 0:
                E2 = torch.stack([E2[:,0], E2[:,1],
                                torch.zeros_like(E2[:,2]),
                                torch.zeros_like(E2[:,3])], dim=-1)
            else:
                if prev is not None:
                    E2 = E2 + prev[idx2]

            pred[idx2] = E2

        if not return_aux:
            return pred

        aux = {
            "gate_logits": gate_logits,
            "chosen_expert": chosen,
        }
        return pred, aux
