import sys
sys.path.append("..")
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from tqdm import tqdm

from app import dataloader
from model import StarNet

parser = argparse.ArgumentParser()
parser.add_argument('--batchsize', type=int, default=8, help='input batch size')
parser.add_argument('--nepoch', type=int, default=8, help='number of epochs to train for')
parser.add_argument('--dataset', type=str, default='../dataset/tiny-imagenet-200/', help="dataset path")
parser.add_argument('--outf', type=str, default='model.pt', help='output folder')
parser.add_argument('--modelf', type=str, default='None', help='model path')
parser.add_argument('--workers', type=int, default=2, help='number of workers to load data')


opt = parser.parse_args()
print(opt)

dataset = 
dataloader = torch.utils.data.DataLoader(dataset,shuffle=True,batch_size=opt.batchsize,num_workers=opt.workers)

val_dataset = 
valdataloader = torch.utils.data.DataLoader(val_dataset,shuffle=True,batch_size=opt.batchsize,num_workers=opt.workers)

print('length of dataset: %s ; length of val dataset: %s' % (len(dataset), len(val_dataset)))
num_batch = len(dataset) / opt.batchsize

model = 
model.cuda()

if opt.modelf != 'None':
    model.load_state_dict(torch.load('../model'+opt.modelf))
    print('Use model from'+opt.modelf)
else:
    print('Use new model')

optimizer = torch.optim.Adam(model.parameters(),lr=0.0001,betas=(0.9, 0.999))

for epoch in tqdm(range(opt.nepoch)):
    show_loss = 0
    for i, data in tqdm(enumerate(dataloader)):
        label,image = data
        image = (image.float())/255.0
        label,image = label.cuda(),image.cuda()
        optimizer.zero_grad()
        model = model.train()
        pred = model(image)
        label_vec = torch.zeros(pred.size())
        for iterate in range(opt.batchsize):
            label_vec[iterate][label[iterate]] = 1
        label_vec = label_vec.cuda()
        loss = F.pairwise_distance(label_vec,pred,p=2)
        loss = loss.sum()
        loss.backward()
        optimizer.step()
        show_loss = show_loss + loss.item()
        if i % 10 == 0:
            # testdata
            print('[ epoch: %d  batch: %d/%d ]  loss: %f' % (epoch,i,num_batch,show_loss,))
            show_loss = 0

torch.save(model.state_dict(),'../model'+opt.outf)
print('Model saved at'+opt.outf)
