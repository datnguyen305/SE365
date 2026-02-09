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
        self.fc1 = nn.Linear(self.input_size, config.hidden_size)
        self.fc2 = nn.Linear(config.hidden_size, self.output_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # input [B, C, H, W] -> flatten -> [B, C*H*W]
        x = x.reshape(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
        
    def predict(self, x):
        logits = self.forward(x)
        preds = torch.argmax(logits, dim=1)
        return preds
        