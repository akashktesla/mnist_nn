import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(28*28,100),
                nn.Linear(100,10))
    def forward(self,x):
        return self.model(x)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
                nn.Conv2d(1, 32, 5),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(32, 64, 5),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Flatten(),
                nn.Linear(4*4*64, 10)
                )
    def forward(self,x):
        return self.model(x)

