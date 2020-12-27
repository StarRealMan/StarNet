import sys
sys.path.append("..")
import os
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm

from app import dataloader
from model import StarNet
from app import visualizer

parser = argparse.ArgumentParser()
parser.add_argument('--batchsize', type=int, default=8, help='input batch size')
parser.add_argument('--pointnum', type=int, default=4096, help='points per room/sample')
parser.add_argument('--subscale', type=float, default=0.5, help='Dataset subsample before training')
parser.add_argument('--nepoch', type=int, default=8, help='number of epochs to train for')
parser.add_argument('--dataset', type=str, default='../data/Stanford3dDataset_v1.2_Aligned_Version', help='dataset path')
parser.add_argument('--outn', type=str, default='Smodel.pt', help='output model name')
parser.add_argument('--model', type=str, default='None', help='history model path')
parser.add_argument('--workers', type=int, default=2, help='number of workers to load data')
parser.add_argument('--testarea', type=int, default=5, help='Area you want to test with (skip while training)')

opt = parser.parse_args()
print(opt)

if opt.batchsize == 1:
    print('Can not use batchsize 1, Change batchsize to 2')
    opt.batchsize = 2
    
train_dataset = dataloader.S3DISDataset(opt.dataset, opt.pointnum, opt.subscale, opt.testarea, split = 'train')
traindataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=opt.batchsize,\
                                              num_workers=opt.workers, drop_last=True)
num_classes = 14

# val_dataset = 
# valdataloader = 

# dataplotter = visualizer.DataPlotter()
writer = SummaryWriter('../log/scene')

print('length of dataset: %s' % (len(train_dataset)))
batch_num = int(len(train_dataset) / opt.batchsize)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# train process
model = StarNet.SceneSegNet(num_classes)

if opt.model != 'None':
    model.load_state_dict(torch.load('../model/'+opt.model))
    print('Use model from ../model/' + opt.model)
else:
    print('Use new model')

if not os.path.exists('../model/Smodel'):
    os.makedirs('../model/Smodel')

model.to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=0.0001,betas=(0.9, 0.999))

for epoch in tqdm(range(opt.nepoch)):
    for i, data in tqdm(enumerate(traindataloader)):
        # out put data size : [BatchSize PointNum PointChannel(XYZRGB)]
        points, label = data
        points = points.transpose(2, 1)
        points, label = points.to(device=device, dtype=torch.float), label.to(device)
        optimizer.zero_grad()
        model = model.train()
        pred = model(points)
        pred = pred.view(-1, num_classes)
        label = label.view(-1)
        loss = F.nll_loss(pred, label)
        loss.backward()
        optimizer.step()
        # testdata
        print('[ epoch: %d/%d  batch: %d/%d ]  loss: %f' % (epoch, opt.nepoch, i+1, batch_num, loss.item()))
        # dataplotter.DataloadY(loss.item())
        # dataplotter.DataPlot()
        writer.add_scalar('training loss', loss.item(), epoch*len(traindataloader)+i)

    if epoch % 50 == 49:
        torch.save(model.state_dict(), '../model/Smodel/epo' + str(epoch) + opt.outn)
        print('Model saved at ../model/Smodel/epo' + str(epoch) + opt.outn)

torch.save(model.state_dict(), '../model/Smodel/final_' + opt.outn)
print('Model saved at ../model/Smodel/final_' + opt.outn)