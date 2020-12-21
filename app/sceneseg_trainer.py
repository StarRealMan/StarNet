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
parser.add_argument('--nepoch', type=int, default=8, help='number of epochs to train for')
parser.add_argument('--dataset', type=str, default='../data/Stanford3dDataset_v1.2_Aligned_Version', help="dataset path")
parser.add_argument('--outf', type=str, default='model.pt', help='output folder')
parser.add_argument('--modelf', type=str, default='None', help='model path')
parser.add_argument('--workers', type=int, default=2, help='number of workers to load data')

opt = parser.parse_args()
print(opt)

train_dataset = dataloader.S3DISDataset(opt.dataset)
traindataloader = torch.utils.data.DataLoader(train_dataset,shuffle=True,batch_size=opt.batchsize,num_workers=opt.workers)
num_classes = 14

# val_dataset = 
# valdataloader = torch.utils.data.DataLoader(val_dataset,shuffle=True,batch_size=opt.batchsize,num_workers=opt.workers)

print('length of dataset: %s' % (len(train_dataset)))
batch_num = len(train_dataset) / opt.batchsize
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# train process
model = StarNet.SceneSegNet(num_classes)

if opt.modelf != 'None':
    model.load_state_dict(torch.load('../model'+opt.modelf))
    print('Use model from'+opt.modelf)
else:
    print('Use new model')

model.to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=0.0001,betas=(0.9, 0.999))

for epoch in tqdm(range(opt.nepoch)):
    show_loss = 0
    for i, data in tqdm(enumerate(traindataloader)):
        # out put data size : [BatchSize PointChannel(XYZRGB) PointNum]
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
        if i % 100 == 0:
            # testdata
            print('[ epoch: %d  batch: %d/%d ]  loss: %f' % (epoch,i,batch_num,show_loss))
            show_loss = 0

torch.save(model.state_dict(),'../model'+opt.outf)
print('Model saved at'+opt.outf)
