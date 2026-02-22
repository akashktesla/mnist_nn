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
        self.conv1 =  nn.Conv2d(1, 32, 5)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 =  nn.Conv2d(32, 64, 5)
        self.pool2 = nn.MaxPool2d(2)
        self.fc = nn.Linear(4*4*64, 10)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
                
    def forward(self,x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.fc(self.flatten(x))
        return x

