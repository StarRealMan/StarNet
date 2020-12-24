import sys
sys.path.append("..")
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
parser.add_argument('--nepoch', type=int, default=8, help='number of epochs to train for')
parser.add_argument('--dataset', type=str, default='../data/SUNRGBD', help='dataset patH')
parser.add_argument('--outn', type=str, default='Fmodel.pt', help='output model name')
parser.add_argument('--model', type=str, default='None', help='history model path')
parser.add_argument('--workers', type=int, default=2, help='number of workers to load data')

opt = parser.parse_args()
print(opt)

# if opt.batchsize == 1:
#     print('Can not use batchsize 1, Change batchsize to 2')
#     opt.batchsize = 2

train_dataset = dataloader.SUNRGBDDataset('../data/SUNRGBD', 'train')
traindataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=opt.batchsize,\
                                              num_workers=opt.workers, drop_last=True)

num_classes = 14

# val_dataset = 
# valdataloader = 

# dataplotter = visualizer.DataPlotter()
writer = SummaryWriter('../log/frame')

print('length of dataset: %s' % (len(train_dataset)))
batch_num = int(len(train_dataset) / opt.batchsize)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# train process
model = StarNet.FrameSegNet(num_classes)

if opt.model != 'None':
    model.load_state_dict(torch.load('../model/'+opt.model))
    print('Use model from ../model/' + opt.model)
else:
    print('Use new model')

model.to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=0.0001,betas=(0.9, 0.999))

for epoch in tqdm(range(opt.nepoch)):
    for i, data in tqdm(enumerate(traindataloader)):
        # out put data size : [BatchSize Channel(RGB or Depth) Image_Height Image_Width]
        rgb, depth, label = data
        rgb, depth, label = rgb.to(device=device, dtype=torch.float), depth.to(device=device, dtype=torch.float),\
                                                                      label.to(device)
        optimizer.zero_grad()
        model = model.train()
        # rgb, depth pre process
        result = torch.cat((rgb,depth), 1)
        pred = model(result)
        pred = pred.transpose(2, 1)
        pred = pred.transpose(3, 2)
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

torch.save(model.state_dict(), '../model/' + opt.outn)
print('Model saved at ../model/' + opt.outn)
