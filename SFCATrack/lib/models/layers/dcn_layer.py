from torchvision.ops import DeformConv2d
import torch
import torch.nn as nn
import torch.nn.functional as F

def token2feature(tokens):
    B,L,D=tokens.shape
    H=W=int(L**0.5)
    x = tokens.permute(0, 2, 1).view(B, D, W, H).contiguous()
    return x

def feature2token(x):
    B,C,W,H = x.shape
    L = W*H
    tokens = x.view(B, C, L).permute(0, 2, 1).contiguous()
    return tokens
class defomableConv(nn.Module):
    def __init__(self, inC, outC, kH = 3, kW = 3, num_deformable_groups = 4):
        super(defomableConv, self).__init__()
        self.num_deformable_groups = num_deformable_groups
        self.inC = inC
        self.outC = outC
        self.kH = kH
        self.kW = kW
        self.offset = nn.Conv2d(inC, num_deformable_groups * 2 * kH * kW, kernel_size=(kH, kW), stride=(1, 1), padding=(1, 1), bias=False)
        self.deform = DeformConv2d(inC, outC, (kH, kW), stride=1, padding=1, groups=num_deformable_groups)

    def forward(self, x):
        offset = self.offset(x)
        out = self.deform(x, offset)
        return out
class defomableConv_offset(nn.Module):
    def __init__(self, inC, outC, kH = 3, kW = 3, num_deformable_groups = 2):
        super(defomableConv_offset, self).__init__()
        self.num_deformable_groups = num_deformable_groups
        self.inC = inC
        self.outC = outC
        self.kH = kH
        self.kW = kW
        self.offset = nn.Conv2d(inC, num_deformable_groups * 2 * kH * kW, kernel_size=(kH, kW), stride=(1, 1), padding=(1, 1), bias=False)
        self.deform = DeformConv2d(inC, outC, (kH, kW), stride=1, padding=1, groups=num_deformable_groups)

    def forward(self, feta3, x):
        offset = self.offset(feta3)
        out = self.deform(x, offset)
        return out
class deformalignblock(nn.Module):
    def __init__(self, inC, outC):
        super(deformalignblock, self).__init__()
        self.deformConv1 = defomableConv(inC=inC*2, outC=outC)
        self.deformConv2 = defomableConv(inC=inC, outC=outC)
        self.deformConv3 = defomableConv(inC=inC, outC=outC)
        self.deformConv4 = defomableConv_offset(inC=inC, outC=outC)

    def forward(self, fr, ft):
        fr = token2feature(fr)
        ft = token2feature(ft)
        cat_feat = torch.cat((fr, ft), dim=1)
        feat1 = self.deformConv1(cat_feat)
        feat2 = self.deformConv2(feat1)
        feat3 = self.deformConv3(feat2)
        aligned_feat = self.deformConv4(feat3, ft)
        aligned_feat = feature2token(aligned_feat)
        return aligned_feat

class dca(nn.Module):
    def __init__(self, in_dim=768, dim=8, xavier_init=False):
        super().__init__()

        self.adapter_down = nn.Linear(in_dim, dim)  
        self.adapter_up = nn.Linear(dim, in_dim)  
        # self.adapter_mid = nn.Linear(dim, dim)

        #nn.init.xavier_uniform_(self.adapter_down.weight)
        # nn.init.zeros_(self.adapter_mid.bias)
        # nn.init.zeros_(self.adapter_mid.weight)
        nn.init.zeros_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        #self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim
        self.dab = deformalignblock(inC=dim, outC=dim)
        

    def forward(self, fr, ft):
        B, N, C = fr.shape
        fr_down = self.adapter_down(fr)   
        fr_down = self.dropout(fr_down)
        ft_down = self.adapter_down(ft)
        ft_down = self.dropout(ft_down)
        f_t_deform = self.dab(fr_down, ft_down)
        f_fusion = f_t_deform + fr_down
        f_fusion = self.adapter_up(f_fusion)
        
        #x_down = self.act(x_down)
        # x_down = self.adapter_mid(x_down)
        #x_down = self.act(x_down)
        
        # x_up = self.adapter_up(x_down)  
        #print("return adap x", x_up.size())
        return f_fusion
class DeformableAlign(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        # 偏移量生成（每个模态独立）
        self.offset_conv = nn.Sequential(
            nn.Conv2d(in_channels, 2*kernel_size**2, kernel_size, padding=1)
        )
        self.deform_conv = DeformConv2d(in_channels, out_channels, kernel_size, padding=1)

    def forward(self, x, y):
        offsets = self.offset_conv(y)
        return self.deform_conv(x, offsets)

class DeformAlignBlock(nn.Module):
    def __init__(self, inC, embedC):
        super().__init__()
        self.deform_align_1 = DeformableAlign(inC, embedC)
        self.deform_align_2 = DeformableAlign(embedC, embedC)
        self.deform_align_3 = DeformableAlign(embedC, inC)
    def forward(self, x1, y):
        x1 = self.deform_align_1(x1,y)
        x1 = self.deform_align_2(x1,x1)
        x1 = self.deform_align_3(x1,x1)
        return x1

class GatedFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(2*channels, channels//2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//2, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, feat1, feat2):
        gate = self.gate(torch.cat([feat1, feat2], dim=1))
        return gate * feat1 + (1 - gate) * feat2
class projection_adapter(nn.Module):
    def __init__(self, in_dim=768, dim=8, xavier_init=False):
        super().__init__()

        self.adapter_down = nn.Linear(in_dim, dim)  
        self.adapter_up = nn.Linear(dim, in_dim)  
        self.adapter_mid = nn.Linear(dim, dim)

        #nn.init.xavier_uniform_(self.adapter_down.weight)


        nn.init.zeros_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)
        nn.init.zeros_(self.adapter_mid.bias)
        nn.init.zeros_(self.adapter_mid.weight)

        #self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim
    def forward(self, x):
        B, N, C = x.shape
        x_down = self.adapter_down(x)   
        
        # x_down = self.act(x_down)
        x_down = self.adapter_mid(x_down)
        x_down = self.dropout(x_down)
        #x_down = self.act(x_down)

        x_up = self.adapter_up(x_down)  
        #print("return adap x", x_up.size())
        return x_up
        
class align_projection_adapter(nn.Module):
    def __init__(self, num_token=256, aggregate_token_num=32, in_dim=768, dim=8, xavier_init=False):
        super().__init__()
        self.rgb_align_adapter = projection_adapter(num_token, aggregate_token_num)
        self.ir_align_adapter = projection_adapter(num_token, aggregate_token_num)
        self.proj_adapter = projection_adapter(in_dim, dim)
    def forward(self, rgb_tokens, tir_tokens):
        rgb_tokens = rgb_tokens.permute(0, 2, 1)
        tir_tokens = tir_tokens.permute(0, 2, 1)
        rgb_tokens = self.rgb_align_adapter(rgb_tokens).permute(0, 2, 1)
        tir_tokens = self.ir_align_adapter(tir_tokens).permute(0, 2, 1)
        rgb_tokens = self.proj_adapter(rgb_tokens)
        tir_tokens = self.proj_adapter(tir_tokens)
        fusion_tokens = self.proj_adapter(rgb_tokens + tir_tokens)
        return rgb_tokens, tir_tokens, fusion_tokens
class MMAF(nn.Module):
    def __init__(self, num_token=256, embed_dim=768):
        # 调用父类的构造函数
        super().__init__()
        # 独立对齐模块
        self.align_rgb = DeformAlignBlock(embed_dim, embed_dim//6)
        self.align_tir = DeformAlignBlock(embed_dim, embed_dim//6)
        self.fusion = GatedFusion(embed_dim)
        self.projection = align_projection_adapter(num_token=num_token, aggregate_token_num=num_token//6, in_dim=embed_dim, dim=embed_dim//6)

    def _reshape_tokens(self, tokens):
        """将ViT输出token转换为空间特征图"""
        b, n, c = tokens.shape
        h = w = int(n**0.5)  # 假设为方形特征
        spatial = tokens.reshape(b, h, w, c).permute(0, 3, 1, 2)
        return spatial

    def _reshape_spatial(self, spatial):
        """将空间特征图转换为ViT输入token"""
        b, c, h, w = spatial.shape
        tokens = spatial.permute(0, 2, 3, 1).reshape(b, h*w, c)
        return tokens

    def forward(self, rgb_tokens, tir_tokens):
        # 模态特征重塑 [B, N, C] → [B, C, H, W]
        rgb_tokens, tir_tokens, fusion_tokens = self.projection(rgb_tokens, tir_tokens)
        rgb_feat = self._reshape_tokens(rgb_tokens)
        tir_feat = self._reshape_tokens(tir_tokens)
        fused_feat = self._reshape_tokens(fusion_tokens)

        # 独立空间对齐
        aligned_rgb = self.align_rgb(rgb_feat, fused_feat)
        aligned_tir = self.align_tir(tir_feat, fused_feat)

        # 门控融合
        fused_feat = self.fusion(aligned_rgb, aligned_tir)
        return self._reshape_spatial(fused_feat)

if __name__ == '__main__':
    fr = torch.rand(2, 256, 768)
    ft = torch.rand(2, 256, 768)
    f_rt = MMAF()(fr, ft)
    print(f_rt.size())