import sys
sys.path.append("..")
import torch
import argparse

from app import dataloader
from model import StarNet
from app import visualizer

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='../data/Stanford3dDataset_v1.2_Aligned_Version', help='dataset path')
parser.add_argument('--model', type=str, default='model.pt', help='history model path')
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

IOU = []

with torch.no_grad():
    for i, data in enumerate(testdataloader):
        rgb, depth, label = data
        rgb, depth = rgb.to(dtype=torch.float), depth.to(dtype=torch.float)
        model = model.eval()
        # rgb, depth pre process
        result = 
        pred = model(result)
        pred = pred.view(-1, num_classes)
        label = label.view(-1)
        
        IOU.append(visualizer.calIOU(pred, label))
        
print('test data iou is as followed:')
print(IOU)