from DoConv import DOConv2d
import torch
import torch.nn as nn
class net(torch.nn.Module):
    def __init__(self, hidden=100, output=10):
        super(net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 2, 1)
        self.pool = nn.AvgPool2d(3, stride=2, padding=1)
        self.conv2 = DOConv2d(20, 50, 5, 1, 1, 0)
        self.fc1 = nn.Linear(200, hidden)
        self.fc2 = nn.Linear(hidden, output)

        self.hidden_size = hidden
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)


    def forward(self, x):
        x = self.conv1(x)
        x = 0.125 * x * x + 0.5 * x + 0.25
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = 0.125 * x * x + 0.5 * x + 0.25
        x = self.fc2(x)
        return x
