import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np

class TransNet(nn.Module):

    def __init__(self):
        super(TransNet, self).__init__()
        self.conv1 = torch.nn.Conv1d(6, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    # input x size : [BatchSize PointChannel(xyzrgb) PointNum]
    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = autograd.Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        
        return x


class SceneSegNet(nn.Module):

    def __init__(self, class_num):
        super(SceneSegNet, self).__init__()

        self.class_num = class_num;
        self.stn = TransNet()
        self.conv1 = torch.nn.Conv1d(6, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.conv4 = torch.nn.Conv1d(1088, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 256, 1)
        self.conv6 = torch.nn.Conv1d(256, 128, 1)
        self.conv7 = torch.nn.Conv1d(128, self.class_num, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.bn6 = nn.BatchNorm1d(128)

    # input x size : [BatchSize PointChannel(xyzrgb) PointNum]
    def forward(self, x):
        
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        cordx = x[:, :, :3]
        rgbx = x[:, :, 3:]
        cordx = torch.bmm(cordx, trans)
        x = torch.cat((cordx,rgbx), 2)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
        x = torch.cat([x, pointfeat], 1)

        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.conv7(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.class_num), dim=-1)
        x = x.view(batchsize, n_pts, self.class_num)
        return x


class FrameSegNet(nn.Module):

    def __init__(self):
        super(FrameSegNet, self).__init__()

    # input x size : [BatchSize ]
    def forward(self, x):
        
        return x



if __name__ == '__main__':
    net = SceneSegNet(14)
    print(net)
    rand = torch.randn(2,6,4096)
    print(net(rand))