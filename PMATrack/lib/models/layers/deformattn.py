import torch
import torch.nn as nn
import torch.nn.functional as F

def homo9_to_H(h, n_heads: int):
    """
    统一把单应输入转换为 (B, n_heads, 3, 3)
    支持输入形状：
      - (B, 9)                 ：所有 head 共享一套 9 参数
      - (B, n_heads, 9)        ：每个 head 各一套 9 参数
      - (B, 3, 3)              ：所有 head 共享一套 3x3
      - (B, n_heads, 3, 3)     ：每个 head 各一套 3x3
    """
    import torch
    if h.dim() == 2 and h.size(-1) == 9:
        # (B, 9) -> (B, n_heads, 3,3)
        B = h.size(0)
        H = torch.zeros(B, n_heads, 3, 3, device=h.device, dtype=h.dtype)
        H[..., 0, 0] = h[..., 0]; H[..., 0, 1] = h[..., 1]; H[..., 0, 2] = h[..., 2]
        H[..., 1, 0] = h[..., 3]; H[..., 1, 1] = h[..., 4]; H[..., 1, 2] = h[..., 5]
        H[..., 2, 0] = h[..., 6]; H[..., 2, 1] = h[..., 7]; H[..., 2, 2] = h[..., 8]
        return H

    if h.dim() == 3 and h.size(-1) == 9:
        # (B, n_heads, 9) -> (B, n_heads, 3,3)
        B, Hh, _ = h.shape
        assert Hh == n_heads, f"Provided heads {Hh} != n_heads {n_heads}"
        H = torch.zeros(B, n_heads, 3, 3, device=h.device, dtype=h.dtype)
        H[..., 0, 0] = h[..., 0]; H[..., 0, 1] = h[..., 1]; H[..., 0, 2] = h[..., 2]
        H[..., 1, 0] = h[..., 3]; H[..., 1, 1] = h[..., 4]; H[..., 1, 2] = h[..., 5]
        H[..., 2, 0] = h[..., 6]; H[..., 2, 1] = h[..., 7]; H[..., 2, 2] = h[..., 8]
        return H

    if h.dim() == 3 and h.shape[-2:] == (3, 3):
        # (B, 3,3) -> (B, n_heads, 3,3)  共享到所有 head
        return h.unsqueeze(1).expand(-1, n_heads, -1, -1).contiguous()

    if h.dim() == 4 and h.shape[-2:] == (3, 3):
        # (B, n_heads, 3,3)
        B, Hh, _, _ = h.shape
        assert Hh == n_heads, f"Provided heads {Hh} != n_heads {n_heads}"
        return h


def build_ref_points(Hf, Wf, device):
    ys, xs = torch.meshgrid(
        torch.arange(Hf, device=device), torch.arange(Wf, device=device), indexing="ij"
    )
    ref_x = (xs + 0.5) / Wf
    ref_y = (ys + 0.5) / Hf
    ref = torch.stack([ref_x, ref_y], dim=-1).reshape(1, Hf*Wf, 2)  # (1, L, 2)
    return ref

def to_grid_norm_xy(x_pix, y_pix, Hf, Wf):
    x = (x_pix / (Wf - 1)) * 2 - 1
    y = (y_pix / (Hf - 1)) * 2 - 1
    return torch.stack([x, y], dim=-1)

class DeformHomographyAttn(nn.Module):
    """
    单尺度 Deformable Attention + Homography 先验（支持 per-head 9参数单应）
    输入:
      q:   (B, L, C)            —— 目标视角查询（token）
      src: (B, C, Hf, Wf)       —— 源视角特征图（被采样）
      h9:  (B, 9) 或 (B, n_heads, 9)  —— 源->目标 单应
    可选:
      H_img/W_img: 若 h9 在原图坐标，自动做到特征图坐标的一致化
    """
    def __init__(self, C, n_heads=8, n_points=4):
        super().__init__()
        assert C % n_heads == 0
        self.C = C
        self.dh = C // n_heads
        self.n_heads = n_heads
        self.n_points = n_points

        self.offset_mlp = nn.Linear(C, n_heads * n_points * 2)  # ΔL
        self.attn_mlp   = nn.Linear(C, n_heads * n_points)      # A
        self.v_proj     = nn.Conv2d(C, C, 1)
        self.out_proj   = nn.Linear(C, C)
        self.delta_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, q, src, h9, H_img=256, W_img=256):
        B, L, C = q.shape
        _, _, Hf, Wf = src.shape
        dev = q.device

        # (1) per-head 单应矩阵
        H_s2t = homo9_to_H(h9, self.n_heads)  # (B, n_heads, 3, 3)

        # (2) 若 h9 在原图坐标 -> 尺度一致化到特征图坐标
        if (H_img is not None) and (W_img is not None):
            # S: (B, 1, 3, 3) 以便广播到 heads
            S = torch.tensor([[Wf/W_img, 0, 0],
                              [0, Hf/H_img, 0],
                              [0, 0, 1.0]], device=dev, dtype=q.dtype).view(1,1,3,3).expand(B,1,3,3)
            S_inv = torch.inverse(S)
            H_s2t = S @ H_s2t @ S_inv  # (B, n_heads, 3, 3)

        # (3) 得到 目标->源
        H_t2s = torch.inverse(H_s2t) # (B, n_heads, 3, 3)

        # (4) 参考点（目标网格中心）与其 [-1,1] 形式
        ref = build_ref_points(Hf, Wf, dev).expand(B, -1, -1)  # (B, L, 2)
        ref_grid = torch.stack([ref[...,0]*2-1, ref[...,1]*2-1], dim=-1)  # (B, L, 2)

        # (5) 计算几何先验 ΔH —— per-head
        x_ref_pix = ref[...,0] * Wf   # (B, L)
        y_ref_pix = ref[...,1] * Hf   # (B, L)
        ones = torch.ones_like(x_ref_pix)
        ref_homo = torch.stack([x_ref_pix, y_ref_pix, ones], dim=-1)            # (B, L, 3)
        ref_homo = ref_homo.unsqueeze(1).expand(-1, self.n_heads, -1, -1)       # (B, heads, L, 3)

        # (B, heads, L, 3) @ (B, heads, 3, 3)^T -> (B, heads, L, 3)
        p_src = torch.matmul(ref_homo, H_t2s.transpose(-1, -2))
        p_src_x = p_src[..., 0] / (p_src[..., 2] + 1e-8)
        p_src_y = p_src[..., 1] / (p_src[..., 2] + 1e-8)
        p_src_grid = to_grid_norm_xy(p_src_x, p_src_y, Hf, Wf)  # (B, heads, L, 2)

        delta_H = p_src_grid - ref_grid.unsqueeze(1)            # (B, heads, L, 2)

        # (6) 可学习微偏移 ΔL 与注意力权重（与原 Deformable DETR 一致）
        delta_L = self.offset_mlp(q).view(B, L, self.n_heads, self.n_points, 2)
        delta_L = self.delta_scale * delta_L                    # (B, L, heads, K, 2)

        attn_w = self.attn_mlp(q).view(B, L, self.n_heads, self.n_points)
        attn_w = F.softmax(attn_w, dim=-1)                      # (B, L, heads, K)

        # (7) 最终采样点：ref_grid + ΔH + ΔL
        #     把 ΔH 调整到 (B, L, heads, 1, 2)
        delta_H_b = delta_H.permute(0,2,1,3).unsqueeze(3)       # (B, L, heads, 1, 2)
        samp_grid = ref_grid.unsqueeze(2).unsqueeze(3) + delta_H_b + delta_L  # (B, L, heads, K, 2)

        # (8) 值特征拆成 heads，并逐点采样
        v = self.v_proj(src).view(B, self.n_heads, self.dh, Hf, Wf)            # (B, heads, dh, Hf, Wf)
        v_flat = v.contiguous().view(B*self.n_heads, self.dh, Hf, Wf)

        # (B, L, heads, K, 2) -> (B*heads, L*K, 1, 2)
        samp_grid_flat = samp_grid.permute(0,2,1,3,4).contiguous().view(B*self.n_heads, L*self.n_points, 1, 2)

        sampled = F.grid_sample(
            v_flat, samp_grid_flat, mode="bilinear", padding_mode="zeros", align_corners=True
        )  # (B*heads, dh, L*K, 1)

        sampled = sampled.squeeze(-1).view(B, self.n_heads, self.dh, L, self.n_points)  # (B,heads,dh,L,K)
        sampled = sampled.permute(0, 3, 1, 4, 2).contiguous()  # (B, L, heads, K, dh)

        out = (attn_w.unsqueeze(-1) * sampled).sum(dim=3)      # (B, L, heads, dh)
        out = out.reshape(B, L, self.C)
        out = self.out_proj(out)
        return out

def build_ref_points2(H, W, device, dtype=torch.float32):
    """
    生成 [0,1] 归一化参考点，按 (y, x) 的行优先顺序展平。
    返回: (L=H*W, 2) -> (x, y) in [0,1]
    """
    ys = torch.linspace(0.5 / H, 1.0 - 0.5 / H, H, device=device, dtype=dtype)  # 像素中心
    xs = torch.linspace(0.5 / W, 1.0 - 0.5 / W, W, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")   # [H,W]
    ref = torch.stack([xx, yy], dim=-1).view(H * W, 2)  # [L,2]
    return ref  # (L,2) in [0,1]


class DeformAttn(nn.Module):
    """
    单尺度 Deformable Attention（无 Homography 先验）
    输入:
      q:   (B, L, C)            —— 目标视角查询（token）
      src: (B, C, Hf, Wf)       —— 源视角特征图（被采样）
      ref_points: (可选) (B, L, 2) —— 每个查询的参考点，[-1,1] 归一化网格坐标
                 若不提供，默认 L 必须等于 Hf*Wf，自动按规则栅格生成参考点
    参数:
      C: 通道数；n_heads: 多头数；n_points: 每头的采样点数
    """
    def __init__(self, C, n_heads=8, n_points=4):
        super().__init__()
        assert C % n_heads == 0, "C must be divisible by n_heads"
        self.C = C
        self.dh = C // n_heads
        self.n_heads = n_heads
        self.n_points = n_points

        # 与 Deformable DETR 一致：从 q 预测 K 个偏移 + 注意力权重
        self.offset_mlp = nn.Linear(C, n_heads * n_points * 2)  # ΔL (grid 空间)
        self.attn_mlp   = nn.Linear(C, n_heads * n_points)      # A (softmax over K)
        self.v_proj     = nn.Conv2d(C, C, 1)
        self.out_proj   = nn.Linear(C, C)

        # 控制初始偏移幅度，训练中可自适应
        self.delta_scale = nn.Parameter(torch.tensor(0.5))

        # 可选：轻微的 LayerNorm 稳定训练（根据需要打开）
        # self.q_norm = nn.LayerNorm(C)

    def forward(self, q, src, ref_points=None):
        """
        q:          (B, L, C)
        src:        (B, C, Hf, Wf)
        ref_points: (可选) (B, L, 2)  归一化到 [-1,1] 的网格坐标
        """
        B, L, C = q.shape
        _, _, Hf, Wf = src.shape

        # 如果未提供 ref_points，要求 L == Hf*Wf，按规则栅格生成
        if ref_points is None:
            if L != Hf * Wf:
                raise ValueError(
                    f"ref_points 未提供时要求 L==Hf*Wf，但得到 L={L}, Hf={Hf}, Wf={Wf}"
                )
            ref = build_ref_points2(Hf, Wf, q.device, q.dtype)     # (L,2) in [0,1]
            # 转为 [-1,1] 的网格坐标
            ref_grid = torch.stack([ref[...,0]*2 - 1, ref[...,1]*2 - 1], dim=-1)  # (L,2)
            ref_grid = ref_grid.unsqueeze(0).expand(B, -1, -1).contiguous()       # (B,L,2)
        else:
            # 直接使用来访坐标（应为 [-1,1]）
            if ref_points.shape != (B, L, 2):
                raise ValueError(f"ref_points 形状应为 (B,L,2)，但得到 {ref_points.shape}")
            ref_grid = ref_points

        # （可选）规范化 q
        # q = self.q_norm(q)

        # 预测偏移 ΔL 与注意力 A
        delta_L = self.offset_mlp(q).view(B, L, self.n_heads, self.n_points, 2)
        delta_L = self.delta_scale * delta_L                         # (B, L, heads, K, 2)

        attn_w = self.attn_mlp(q).view(B, L, self.n_heads, self.n_points)
        attn_w = F.softmax(attn_w, dim=-1)                           # (B, L, heads, K)

        # 最终采样点 = ref_grid + ΔL （都在 [-1,1] 归一化网格空间）
        samp_grid = ref_grid.unsqueeze(2).unsqueeze(3) + delta_L     # (B, L, heads, K, 2)

        # 值特征分头
        v = self.v_proj(src).view(B, self.n_heads, self.dh, Hf, Wf)  # (B, heads, dh, Hf, Wf)
        v_flat = v.contiguous().view(B * self.n_heads, self.dh, Hf, Wf)

        # grid_sample 需要 (N, Hout, Wout, 2)；这里我们按 “每个 token 有 K 个点” 来采样，
        # 把它当成 Hout = L*K, Wout = 1 的特殊“图像”采样
        samp_grid_flat = samp_grid.permute(0, 2, 1, 3, 4).contiguous()            # (B, heads, L, K, 2)
        samp_grid_flat = samp_grid_flat.view(B * self.n_heads, L * self.n_points, 1, 2)

        sampled = F.grid_sample(
            v_flat, samp_grid_flat, mode="bilinear", padding_mode="zeros", align_corners=True
        )  # (B*heads, dh, L*K, 1)

        # 复原形状并按注意力聚合 K 个采样点
        sampled = sampled.squeeze(-1).view(B, self.n_heads, self.dh, L, self.n_points)  # (B,heads,dh,L,K)
        sampled = sampled.permute(0, 3, 1, 4, 2).contiguous()  # (B, L, heads, K, dh)

        out = (attn_w.unsqueeze(-1) * sampled).sum(dim=3)      # (B, L, heads, dh)
        out = out.reshape(B, L, self.C)
        out = self.out_proj(out)
        return out
