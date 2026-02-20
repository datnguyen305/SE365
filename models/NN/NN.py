import torch 
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.w = config.width
        self.h = config.height
        self.c = config.channels
        self.input_size = self.w * self.h * self.c
        self.output_size = kwargs.get("labels", 2)
        
        # Giả sử bạn muốn dùng cùng một hidden_size cho cả 3 lớp
        hidden = config.hidden_size
        
        # Khai báo 3 lớp ẩn
        self.fc1 = nn.Linear(self.input_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        
        # Lớp đầu ra (output layer)
        self.fc_out = nn.Linear(hidden, self.output_size)
        
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # Flatten: [B, C, H, W] -> [B, C*H*W]
        x = x.reshape(x.size(0), -1)
        
        # Lớp 1
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Lớp 2
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Lớp 3
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        
        # Lớp đầu ra
        x = self.fc_out(x)
        return x
        
    def predict(self, x):
        logits = self.forward(x)
        preds = torch.argmax(logits, dim=1)
        return preds