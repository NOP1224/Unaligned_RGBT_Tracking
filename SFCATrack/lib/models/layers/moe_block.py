import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50
from torch.nn import functional as F

class ResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet18(pretrained=False)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Identity()

    def forward(self, x):
        return self.model(x)  # (B, 512)

class SharedExpert(nn.Module):
    """带残差连接的专家网络"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.w1 = nn.Linear(input_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, output_dim)
        self.w3 = nn.Linear(input_dim, hidden_dim)
    
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) + self.w3(x))

class RouterExpert(nn.Module):
    """带残差连接的专家网络"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.w1 = nn.Linear(input_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, output_dim)
        self.w3 = nn.Linear(input_dim, hidden_dim)
    
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) + self.w3(x))

class Router(nn.Module):
    """路由网络"""
    def __init__(self, input_dim, hidden_dim, num_experts):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_experts),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        return self.net(x) 
    
class MLP_Bias_Head(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers = 3, BN=False):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim//2] * (num_layers - 1)
        self.tanh = nn.Tanh()
        if BN:
            self.layers = nn.ModuleList(nn.Sequential(nn.Linear(n, k), nn.BatchNorm1d(k))
                                        for n, k in zip([input_dim] + h, h + [output_dim]))
        else:
            self.layers = nn.ModuleList(nn.Linear(n, k)
                                        for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else self.tanh(layer(x))  # 最后一层使用tanh激活函数
        return x

class MoE(nn.Module):
    """改进的MoE模块"""
    def __init__(self, input_dim=1024, num_experts=10, hidden_dim=512, bias_output=4):
        super().__init__()
        self.num_experts = num_experts
        self.n_activated_experts = 1  # 初始激活一个专家
        
        # 专家网络
        self.shared_expert = SharedExpert(input_dim, hidden_dim, input_dim)
        self.experts = nn.ModuleList([
            RouterExpert(input_dim, hidden_dim, input_dim) for _ in range(num_experts-1)
        ])
        
        # 路由网络（使用温度系数调节softmax）
        self.router = Router(input_dim, hidden_dim, num_experts)
        
        self.mlp_head = MLP_Bias_Head(input_dim, hidden_dim, bias_output)
    def forward_phase1(self, x):
        # 第一阶段：只使用共享专家
        x = self.shared_expert(x) + x
        out = self.mlp_head(x)
        return out

    def forward_phase2(self, x, expert_index):
        # 第二阶段：冻结共享专家，逐个训练其他专家
        shared_out = self.shared_expert(x)
        expert_out = self.experts[expert_index](x)
        out = shared_out + expert_out + x
        out = self.mlp_head(out)
        return out

    def forward(self, x):
        # 路由权重生成
        weights = self.router(x)  # (B, num_experts)
        
        # 强制共享专家激活 + top-1选择
        shared_weights = weights[:, 0:1]  # (B, 1)
        other_weights = weights[:, 1:]    # (B, num_experts-1)
        
        # 对其他专家取top-1
        top1_values, top1_indices = torch.topk(other_weights, k=1, dim=1)
        
        # 创建掩码
        mask = torch.zeros_like(other_weights)
        mask.scatter_(1, top1_indices, 1.0)
        masked_weights = other_weights * mask
        
        # 合并权重并重新归一化
        new_weights = torch.cat([shared_weights, masked_weights], dim=1)
        new_weights = new_weights / (new_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        # 专家输出计算
        shared_out = self.shared_expert(x)
        expert_outs = [expert(x) for expert in self.experts]
        all_outs = torch.stack([shared_out] + expert_outs, dim=1)  # (B, 12, D_out)
        
        # 加权融合
        combined_out = torch.einsum('be,beo->bo', new_weights, all_outs)
        
        combined_out = combined_out + x
        
        out = self.mlp_head(combined_out)
        
        return out

class BiasNetwork(nn.Module):
    """完整网络架构"""
    def __init__(self):
        super().__init__()
        self.feature_extractor = ResNetBackbone()
        self.moe = MoE()
        

    def forward(self, x, training_phase=None, expert_index=None):
        features_rgb = self.feature_extractor(x[:,:3,:,:])
        features_ir = self.feature_extractor(x[:,3:,:,:])
        features = torch.cat([features_rgb, features_ir], dim=1)
        if not self.training:
            if training_phase == 'phase1':
                return self.moe.forward_phase1(features)
            else:
                return self.moe(features)
        else:
            # print("training...")
            if training_phase == 'att_moe_phase1':
                # print("phase 1")
                return self.moe.forward_phase1(features)
            elif training_phase == 'att_moe_phase2':
                return self.moe.forward_phase2(features, expert_index=expert_index)
            else:
                # print("phase 3")
                return self.moe(features)

# 验证实现
if __name__ == "__main__":
    training_phase = 'att_moe_phase1'
    expert_index = 0
    model = BiasNetwork()
    dummy_input = torch.randn(4, 6, 512, 512)
    output = model(dummy_input, training_phase, expert_index)
    
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出范围验证: 最大值={output.max().item():.4f}, 最小值={output.min().item():.4f}")
    print(output)