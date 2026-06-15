import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.layers.slove import warp_feature_inverse_grid
from ...train.actors.homography_compute import HomographyNormalizer, SimpleHomoNormalizer
# ===============================
# 偏移量统一转换到 Homography
# ===============================
def offset_to_homography(offset, device):
    B = offset.shape[0]

    dx, dy, sx, sy = offset[:,0], offset[:,1], offset[:,2], offset[:,3]
    H = torch.eye(3, device=device).unsqueeze(0).repeat(B,1,1)
    H[:,0,0] = sx
    H[:,1,1] = sy
    H[:,0,2] = dx
    H[:,1,2] = dy
    
    return H

# -----------------------
# 生成 Warp 网格位置编码
# -----------------------
def homography_warp_grid(H, Ht, Wt, device):
    """ 
    H: [B, 3, 3] 单应性矩阵
    Ht, Wt: 特征图高宽
    返回: warp 后的网格位置编码 [B, Ht, Wt, 2]
    """
    B = H.size(0)
    # 标准归一化坐标网格 [-1, 1]
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, Ht, device=device),
        torch.linspace(-1, 1, Wt, device=device),
        indexing="ij"
    )
    grid = torch.stack([x, y, torch.ones_like(x)], dim=-1)  # [Ht, Wt, 3]
    grid = grid.view(-1, 3).t().unsqueeze(0).repeat(B, 1, 1)  # [B, 3, Ht*Wt]

    # warp
    warped = H @ grid  # [B, 3, Ht*Wt]
    warped = warped[:, :2, :] / warped[:, 2:3, :]
    warped = warped.transpose(1, 2).view(B, Ht, Wt, 2)  # [B, Ht, Wt, 2]
    return warped


# -----------------------
# 轻量卷积模块
# -----------------------
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


# ===============================
# 改造后的 Refiner
# ===============================
class WarpGridRefinerLight(nn.Module):
    def __init__(self, in_ch, hidden_ch, num_heads=2, pre_warp=False):
        super().__init__()
        self.enc1 = DepthwiseSeparableConv(in_ch, hidden_ch)
        self.enc2 = DepthwiseSeparableConv(hidden_ch, hidden_ch)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_ch, num_heads=num_heads, batch_first=True)
        self.dec1 = DepthwiseSeparableConv(hidden_ch, hidden_ch)
        self.dec2 = DepthwiseSeparableConv(hidden_ch, in_ch)
        self.pre_warp = pre_warp
    def forward(self, feat, offset):
        """
        feat: [B, C, H, W]
        offset: shift[2] / scale[4] / warp[3,3]
        offset_type: "shift" | "scale" | "warp"
        """
        B, C, Ht, Wt = feat.shape

        # 统一转成 Homography
        H_mat = offset_to_homography(offset, feat.device)
        # 如果需要先 warp
        if self.pre_warp:
            feat, valid_mask = warp_feature_inverse_grid(feat, H_mat, out_size=(Ht, Wt), padding_mode='zeros')

        # 生成 warp 网格位置编码
        warp_grid = homography_warp_grid(H_mat, Ht, Wt, feat.device)  # [B,H,W,2]
        warp_pe = warp_grid.permute(0, 3, 1, 2)  # [B,2,H,W]

        # 映射到通道维度
        warp_pe = F.interpolate(warp_pe, size=(Ht, Wt), mode="bilinear", align_corners=False)
        warp_pe = F.conv2d(warp_pe, weight=torch.randn(C, 2, 1, 1, device=feat.device))

        # 条件注入
        x = feat + warp_pe

        # Encoder
        x = self.enc1(x)
        x = self.enc2(x)

        # Attention 融合
        x_flat = x.flatten(2).transpose(1, 2)  # [B, HW, C]
        x_attn, _ = self.attn(x_flat, x_flat, x_flat)
        x = x_attn.transpose(1, 2).view(B, -1, Ht, Wt)

        # Decoder
        x = self.dec1(x)
        x = self.dec2(x)
        return x

# class NoParam_Feature_Shifter(nn.Module):
#     def __init__(self, H: int, W: int):
#         super().__init__()
#         self.H, self.W = H, W
#         self.homo_norm = SimpleHomoNormalizer(256, 256)
        
#     def forward(self, feat: torch.Tensor, offset: torch.Tensor, inverse: bool=False):
#         B, H, W, D = feat.shape
#         feat_map = feat.permute(0,3,1,2)  
#         H_mat = offset_to_homography(offset, feat.device)
#         H_mat = self.homo_norm.denormalize(H_mat, inverse=inverse)
#         feat_shifted, _ = warp_feature_inverse_grid(feat_map, H_mat, out_size=(self.H,self.W), padding_mode='zeros')
#         feat_shifted = feat_shifted.permute(0,2,3,1).reshape(B, -1, D)  # [B,HW,D]
#         return feat_shifted


class NoParam_Feature_Shifter(nn.Module):
    def __init__(self, H: int, W: int):
        super().__init__()
        self.H, self.W = H, W
        # 如果 H/ W 一直等于 256，可保留；否则建议改成 (self.W, self.H)
        self.homo_norm = SimpleHomoNormalizer(256, 256)

    def _adapt_H_to_feat_grid(self, H_img: torch.Tensor) -> torch.Tensor:
        """把以 256×256 像素为基准的 H，转换到 (self.W,self.H) 特征图坐标系"""
        B = H_img.shape[0]
        sx = self.W / 256.0
        sy = self.H / 256.0
        S  = H_img.new_zeros(B, 3, 3)
        S[:, 0, 0] = sx; S[:, 1, 1] = sy; S[:, 2, 2] = 1.0
        S_inv = H_img.new_zeros(B, 3, 3)
        S_inv[:, 0, 0] = 1.0/sx; S_inv[:, 1, 1] = 1.0/sy; S_inv[:, 2, 2] = 1.0
        return S.bmm(H_img).bmm(S_inv)  # H_feat = S * H_img * S^{-1}

    def forward(self, feat: torch.Tensor, offset: torch.Tensor, inverse: bool=False):
        """
        feat:  [B, H, W, D]  (这里的 H,W 是特征图分辨率)
        offset: 组装单应性的参数（约定生成 source->target）
        inverse: 仅传给 denormalize()，不要和 warp_feature_inverse_grid 的“反采样”混淆
        """
        B, H, W, D = feat.shape
        feat_map = feat.permute(0, 3, 1, 2).contiguous()  # [B,C,H,W]

        # 1) 由 offset 得到“像素坐标系下”的单应性（以 256×256 为基准）
        H_img = offset_to_homography(offset, feat.device)           # [B,3,3], source->target
        H_img = self.homo_norm.denormalize(H_img, inverse=inverse)  # 仍以 256×256 为基准

        # 2) 若特征图尺寸 != 256×256，则把 H 映射到特征图坐标系
        # if (self.H != 256) or (self.W != 256):
        #     H_mat = self._adapt_H_to_feat_grid(H_img)
        # else:
        H_mat = H_img

        # 3) 进行 inverse-warp 采样（函数内部会对 H 做一次 inverse）
        warped, _ = warp_feature_inverse_grid(
            feat_map, H_mat, out_size=(self.H, self.W), padding_mode='zeros'
        )

        # 4) 回到 [B,HW,D]
        feat_shifted = warped.permute(0, 2, 3, 1).reshape(B, -1, D)
        return feat_shifted