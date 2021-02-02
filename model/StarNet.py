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

class Conv_BN(nn.Module):
    def __init__(self, in_chans, out_chans, padding, batch_norm):
        super(Conv_BN, self).__init__()
        self.batch_norm = batch_norm
        self.myConv1 = nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=padding)
        self.myBN1 = nn.BatchNorm2d(out_chans)
        self.myConv2 = nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=padding)
        self.myBN2 = nn.BatchNorm2d(out_chans)
        
    def forward(self, x):
        x = self.myConv1(x)
        x = F.relu(x)
        if self.batch_norm:
            x = self.myBN1(x)
        x = self.myConv2(x)
        x = F.relu(x)
        if self.batch_norm:
            x = self.myBN2(x)

        return x

class Deconv_Upsample(nn.Module):
    def __init__(self, in_chans, out_chans, padding, batch_norm):
        super(Deconv_Upsample, self).__init__()
        self.myDeconv = nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2)
        self.myConv_BN = Conv_BN(in_chans, out_chans, padding, batch_norm)

    def crop(self, layer, target_size):
        _,_,layer_height, layer_width = layer.shape
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y: (diff_y + target_size[0]), diff_x: (diff_x + target_size[1])]

    def forward(self, x, bridge):
        x = self.myDeconv(x)
        crop = self.crop(bridge, x.shape[2:])
        x = torch.cat([x, crop], 1)
        x = self.myConv_BN(x)
        return x



class FrameSegNet(nn.Module):
    def __init__(self,in_channels=4, n_classes=14, padding=False, batch_norm=False):
        super(FrameSegNet, self).__init__()
        self.padding = padding

        self.Conv_BN1 = Conv_BN(in_channels, 128, padding, batch_norm)
        self.Conv_BN2 = Conv_BN(128, 256, padding, batch_norm)
        self.Conv_BN3 = Conv_BN(256, 512, padding, batch_norm)
        self.Conv_BN4 = Conv_BN(512, 1024, padding, batch_norm)

        self.Deconv1 = Deconv_Upsample(1024, 512, padding, batch_norm)
        self.Deconv2 = Deconv_Upsample(512, 256, padding, batch_norm)
        self.Deconv3 = Deconv_Upsample(256, 128, padding, batch_norm)

        self.Last_Conv = nn.Conv2d(128, n_classes, kernel_size=1)
        
    def forward(self, inputs):
        x1 = self.Conv_BN1(inputs)      # input = 320 * 240
        x = F.max_pool2d(x1 ,2)
        x2 = self.Conv_BN2(x)
        x = F.max_pool2d(x2 ,2)
        x3 = self.Conv_BN3(x)
        x = F.max_pool2d(x3 ,2)
        x = self.Conv_BN4(x)
        x = self.Deconv1(x, x3)
        x = self.Deconv2(x, x2)
        x = self.Deconv3(x, x1)
        x = self.Last_Conv(x)
        x = F.log_softmax(x, 1)         # output = 228 * 148
        
        return x

if __name__ == '__main__':

    # net = SceneSegNet(14)
    # print(net)
    # rand = torch.randn(2, 6, 4096)
    # result = net(rand)
    # print(result)
    # print(result.shape)

    net = FrameSegNet()
    print(net)
    rand = torch.randn(2, 4, 240, 320)
    output = net(rand)
    
    print(output.shape)
    