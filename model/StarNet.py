import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np

class TransNet(nn.Module):

    def __init__(self):
        super(TransNet, self).__init__()
        self.conv1 = nn.Conv1d(6, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
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

        iden = autograd.Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32)))\
                                .view(1,9).repeat(batchsize,1)
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
        self.conv1 = nn.Conv1d(6, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.conv4 = nn.Conv1d(1088, 512, 1)
        self.conv5 = nn.Conv1d(512, 256, 1)
        self.conv6 = nn.Conv1d(256, 128, 1)
        self.conv7 = nn.Conv1d(128, self.class_num, 1)

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

    # onput x size : [BatchSize PointNum ClassNum]
        return x


class FrameSegNet(nn.Module):

    def __init__(self, class_num):
        super(FrameSegNet, self).__init__()

        self.class_num = class_num
        self.conv1_1 = nn.Conv2d(4, 64, 3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)

        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_3 = nn.BatchNorm2d(512)
        
        self.conv6_1 = nn.Conv2d(512, self.class_num, 1)

        self.conv6_2 = nn.Conv2d(512, self.class_num, 1)
        self.deconv1 = nn.ConvTranspose2d(self.class_num, self.class_num, 4, stride=2, padding=1, bias=False)

        self.conv6_3 = nn.Conv2d(256, self.class_num, 1)
        self.deconv2 = nn.ConvTranspose2d(self.class_num, self.class_num, 4, stride=2, padding=1, bias=False)

        self.deconv3 = nn.ConvTranspose2d(self.class_num, self.class_num, 8, stride=4, padding=2, bias=False)

        self.pool = nn.MaxPool2d(2,2)
        self.relu = nn.ReLU(inplace=True)

        


    # input x size : [BatchSize Channel(Processed: 4) Image_Height Image_Width]
    def forward(self, x):

        x = self.relu(self.bn1_1(self.conv1_1(x )))       
        x = self.relu(self.bn1_2(self.conv1_2(x )))
                                                        # Image Size : 240 x 320, channel : 64
        x = self.relu(self.bn2_1(self.conv2_1(x )))
        x = self.relu(self.bn2_2(self.conv2_2(x )))
        x = self.pool(x)                                # Image Size : 120 x 160, channel : 128

        x = self.relu(self.bn3_1(self.conv3_1(x)))
        x = self.relu(self.bn3_2(self.conv3_2(x)))
        x = self.pool(x)
        xlevel1_4 = x                                   # Image Size : 60 x 80, channel : 256

        x = self.relu(self.bn4_1(self.conv4_1(x)))
        x = self.relu(self.bn4_2(self.conv4_2(x)))
        x = self.pool(x)
        xlevel1_8 = x                                   # Image Size : 30 x 40, channel : 512

        x = self.relu(self.bn5_1(self.conv5_1(x)))
        x = self.relu(self.bn5_2(self.conv5_2(x)))
        x = self.relu(self.bn5_3(self.conv5_3(x)))
        x = self.pool(x)
        xlevel1_16 = x                                  # Image Size : 15 x 20, channel : 512

        x = self.conv6_1(xlevel1_16)

        xlevel1_8 = self.conv6_2(xlevel1_8)
        x = self.relu(self.deconv1(x)) + xlevel1_8      # Image Size : 30 x 40, channel : num_class

        xlevel1_4 = self.conv6_3(xlevel1_4)
        x = self.relu(self.deconv2(x)) + xlevel1_4      # Image Size : 60 x 80, channel : num_class

        x = self.deconv3(x)                             # Image Size : 240 x 320, channel : num_class

        x = F.log_softmax(x, dim=1)

    # onput x size : [BatchSize ClassNum Image_Height Image_Width]
        return x


if __name__ == '__main__':

    # net = SceneSegNet(14)
    # print(net)
    # rand = torch.randn(2, 6, 4096)
    # result = net(rand)
    # print(result)
    # print(result.shape)

    net = FrameSegNet(14)
    print(net)
    rand = torch.randn(2, 4, 240, 320)
    result = net(rand)
    print(result)
    print(result.shape)
    