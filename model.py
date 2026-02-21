import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Flatten(),nn.Linear(28*28,100), nn.Linear(100,10))
    def forward(self,x):
        return self.model(x)
