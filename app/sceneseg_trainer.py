import sys
sys.path.append("..")
import torch
import torch.nn.functional as F
import argparse
from tqdm import tqdm

from app import dataloader
from model import StarNet

parser = argparse.ArgumentParser()
parser.add_argument('--batchsize', type=int, default=8, help='input batch size')
parser.add_argument('--pointnum', type=int, default=4096, help='points per room/sample')
parser.add_argument('--nepoch', type=int, default=8, help='number of epochs to train for')
parser.add_argument('--dataset', type=str, default='../data/Stanford3dDataset_v1.2_Aligned_Version', help="dataset path")
parser.add_argument('--outn', type=str, default='model.pt', help='output model name')
parser.add_argument('--model', type=str, default='None', help='history model path')
parser.add_argument('--workers', type=int, default=2, help='number of workers to load data')

opt = parser.parse_args()
print(opt)

if opt.batchsize == 1:
    print('Can not use batchsize 1, Change batchsize to 2')
    opt.batchsize = 2
    
train_dataset = dataloader.S3DISDataset(opt.dataset, opt.pointnum)
traindataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=opt.batchsize,\
                                              num_workers=opt.workers, drop_last=True)
num_classes = 14

# val_dataset = 
# valdataloader = torch.utils.data.DataLoader(val_dataset,shuffle=True,batch_size=opt.batchsize,num_workers=opt.workers)

print('length of dataset: %s' % (len(train_dataset)))
batch_num = int(len(train_dataset) / opt.batchsize)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# train process
model = StarNet.SceneSegNet(num_classes)

if opt.model != 'None':
    model.load_state_dict(torch.load('../model'+opt.model))
    print('Use model from'+opt.model)
else:
    print('Use new model')

model.to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=0.0001,betas=(0.9, 0.999))

for epoch in range(opt.nepoch):
    show_loss = 0
    for i, data in enumerate(traindataloader):
        # out put data size : [BatchSize PointNum PointChannel(XYZRGB)]
        points, label = data
        points = points.transpose(2, 1)
        label,points = label.to(device),points.to(device=device, dtype=torch.float)
        optimizer.zero_grad()
        model = model.train()
        pred = model(points)
        pred = pred.view(-1, num_classes)
        label = label.view(-1, 1)[:, 0]
        loss = F.nll_loss(pred, label)
        loss.backward()
        optimizer.step()
        show_loss = show_loss + loss.item()
        if i % 10 == 9:
            # testdata
            print('[ epoch: %d/%d  batch: %d/%d ]  loss: %f' % (epoch, opt.nepoch, i, batch_num, show_loss))
            show_loss = 0

torch.save(model.state_dict(), '../model' + opt.outn)
print('Model saved at../model' + opt.outn)
