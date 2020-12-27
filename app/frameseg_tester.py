import sys
sys.path.append("..")
import os
import torch
import argparse
import cv2

from app import dataloader
from model import StarNet
from app import visualizer

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='../data/SUNRGBD', help='dataset path')
parser.add_argument('--model', type=str, default='Fmodel/final_Fmodel.pt', help='history model path')
parser.add_argument('--workers', type=int, default=2, help='number of workers to load data')

opt = parser.parse_args()
print(opt)

test_dataset = dataloader.SUNRGBDDataset('../data/SUNRGBD', 'test')
testdataloader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=1,\
                                              num_workers=opt.workers, drop_last=False)

dataset_len = len(test_dataset)
num_classes = 14

model = StarNet.FrameSegNet(num_classes)
model.load_state_dict(torch.load('../model/'+opt.model))
print('Use model from ../model/' + opt.model)

if not os.path.exists('../data/savings/F_TEST/'):
    os.makedirs('../data/savings/F_TEST/')

IOU = []
with torch.no_grad():
    for i, data in enumerate(testdataloader):
        rgb, depth, label = data
        rgb, depth = rgb.to(dtype=torch.float), depth.to(dtype=torch.float)
        rgb = rgb/255
        depth = depth/255
        model = model.eval()
        # rgb, depth pre process
        result = torch.cat((rgb,depth), 1)
        pred = model(result)
        ioupred = pred.view(-1, num_classes)
        ioupred = torch.max(ioupred, 1)[1]
        label = label.view(-1)
        IOU.append(visualizer.calIOU(ioupred, label))
        
        pred = torch.max(pred, 1)[1]
        pred = pred.squeeze(0)
        cv2.imwrite('../data/savings/F_TEST/' + str(i) + '_test.jpg', pred.numpy())
        print('saving visualization file ' + str(i) + '_test.jpg')
        
print('test data iou is as followed:')
print(IOU)