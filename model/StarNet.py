import torch
import torch.nn as nn
import torch.nn.functional as F

class SceneSegNet(nn.Module):

    def __init__(self):
        super(SceneSegNet, self).__init__()

    # input x size : [BatchSize PointNum PointChannel(xyzrgb)]
    def forward(self, x):
        
        return x

class FrameSegNet(nn.Module):

    def __init__(self):
        super(FrameSegNet, self).__init__()

    def forward(self, x):
        
        return x



if __name__ == '__main__':
    net = SceneSegNet()
    print(net)
    rand = torch.randn(1,10,6)
    print(net(rand))