import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Perceptron(nn.Module):
    def __init__(self,  input_dim, output_dim, hidden_dim = 8):
        super(Perceptron, self).__init__()
        self.linear_down = nn.Linear(input_dim, hidden_dim)
        self.linear_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.linear_up = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear_up(self.relu(self.linear_hidden(self.relu(self.linear_down(x))))))



def combined_loss(c, b, lambda_value=0.5):
    # 余弦相似度损失
    c_flat = c.view(-1, c.shape[-1])
    b_flat = b.view(-1, b.shape[-1])
    positive_sim = F.cosine_similarity(c_flat, b_flat, dim=1)
    loss_posi_cont = torch.mean((1 - positive_sim)**2)
    negative_sim = F.cosine_similarity(c_flat, c_flat, dim=-1)
    negative_sim = negative_sim.reshape(c.shape[0], c.shape[1])
    negative_sim = torch.mean(negative_sim, dim=1)
    print(negative_sim)
    
            
        

    return loss_cos + lambda_value * loss_cont


if __name__ == '__main__':
    
    source_tensor = torch.randn(4,16,768)
    target_tensor = torch.randn(4,16,768)
    loss = combined_loss(source_tensor, target_tensor)
    print(loss)
    '''device = torch.device("cuda:3")
    model = Perceptron(16, 16).to(device)

    input_tensor = torch.randn(4,16,768).to(device)
    target_tensor = torch.randn(4,16,768).to(device)


    learning_rate = 0.1

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    num_epochs = 100
    for epoch in range(num_epochs):
        output_tensor = model(input_tensor.permute(0,2,1)).permute(0,2,1)
        loss = contrastive_variant_loss(output_tensor, target_tensor, margin=0.5).to(device)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 2 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')'''
