import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

    # input x size : [BatchSize PointNum PointChannel(xyzrgb)]
    def forward(self, x):
        
        return x


if __name__ == '__main__':
    net = Net()
    print(net)
    rand = torch.randn(1,10,6)
    print(net(rand))