import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.layers.moe import MoENet

def feature2token(x):
    x = x.view(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    return x

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
                 
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_q, x_kv, mask=None, return_attention=False):
        # x: B, N, C
        # mask: [B, N, ] torch.bool
        B_, N_, C_ = x_q.shape
        q = self.q(x_q).reshape(B_, N_, 1, self.num_heads, C_ // self.num_heads).permute(2, 0, 3, 1, 4)
        
        B,N,C = x_kv.shape
        kv = self.kv(x_kv).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale


        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'),)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N_, C_)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attn
        else:
            return x
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act(out)

        return out

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, 
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.act = nn.GELU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * BasicBlock.expansion,
                         kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
                )
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)


        return x
class featurebackbone(nn.Module):
    def __init__(self):
        super(featurebackbone, self).__init__()
        self.resnet1= ResNet18()
        self.resnet2= ResNet18()

    def forward(self, temp, temp_t, x_prior, xi_prior, x_t, xi_t):
        temp = self.resnet1(temp)
        temp_t = self.resnet2(temp_t)
        x_prior = self.resnet1(x_prior)
        xi_prior = self.resnet2(xi_prior)
        x_t = self.resnet1(x_t)
        xi_t = self.resnet2(xi_t)
        temp = feature2token(temp)
        temp_t = feature2token(temp_t)
        x_prior = feature2token(x_prior)
        xi_prior = feature2token(xi_prior)
        x_t = feature2token(x_t)
        xi_t = feature2token(xi_t)
        return x_prior, xi_prior, x_t, xi_t, temp, temp_t
class HardAlignNet(nn.Module):
    def __init__(self, dim = 512):
        super(HardAlignNet, self).__init__()
        self.featurebackbone = featurebackbone()
        self.template_fusion = nn.Linear(dim*2, dim)
        self.attn_t = CrossAttention(dim=dim)
        self.attn_prior = CrossAttention(dim=dim)
        self.attn_gate = nn.MultiheadAttention(embed_dim=dim, num_heads=8)
        self.moe_bias_net = MoENet()

    def forward(self, temp, temp_t, x_prior, xi_prior, x_t, xi_t):
        #提取特征
        x_prior, xi_prior, x_t, xi_t, temp, temp_t = self.featurebackbone(temp, temp_t, x_prior, xi_prior, x_t, xi_t)
        
        #模板特征融合
        temp_fusion = self.template_fusion(torch.cat([temp, temp_t], dim=2))
        
        #交叉注意力关注目标
        f_prior = torch.cat([x_prior, xi_prior], dim=1)
        f_current = torch.cat([x_t, xi_t], dim=1)
        f_prior = self.attn_prior(f_prior, temp_fusion)
        f_current = self.attn_t(f_current, temp_fusion)
        
        #MoE
        f_prior_gate = x_prior - xi_prior
        f_current_gate = x_t - xi_t
        x_gate = torch.cat([f_prior, f_current], dim=1)
        f_gate = self.attn_gate(x_gate, x_gate, x_gate)[0]
        
        if not self.training:
            bias = self.moe_bias_net(f_prior, f_prior_gate, f_current, f_current_gate, f_gate)
            return bias
        else:
            bias, gate_probs, gate_scores1, gate_scores2 = self.moe_bias_net(f_prior, f_prior_gate, f_current, f_current_gate, f_gate)
            return bias, gate_probs, gate_scores1, gate_scores2