import sys
sys.path.append("..")
import os
import torch
import argparse

from app import dataloader
from model import StarNet
from app import visualizer

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='../data/Stanford3dDataset_v1.2_Aligned_Version', help='')
parser.add_argument('--model', type=str, default='Smodel/final_Smodel.pt', help='history model path')
parser.add_argument('--pointnum', type=int, default=4096, help='points per room/sample')
parser.add_argument('--subscale', type=float, default=0.5, help='Dataset subsample before training')
parser.add_argument('--outn', type=str, default='test.pcd', help='output file name')
parser.add_argument('--workers', type=int, default=2, help='number of workers to load data')
parser.add_argument('--testarea', type=int, default=5, help='Area you want to test with (skip while training)')

opt = parser.parse_args()
print(opt)

test_dataset = dataloader.S3DISDataset(opt.dataset, opt.pointnum, opt.subscale, opt.testarea, split = 'test')
testdataloader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size = 1,\
                                              num_workers=opt.workers, drop_last=False)

dataset_len = len(test_dataset)
num_classes = 14

model = StarNet.SceneSegNet(num_classes)
model.load_state_dict(torch.load('../model/'+opt.model))
print('Use model from ../model/' + opt.model)

if not os.path.exists('../data/savings/'+str(opt.testarea)+'_TEST/'):
    os.makedirs('../data/savings/'+str(opt.testarea)+'_TEST/')

IOU = []
with torch.no_grad():
    for i, data in enumerate(testdataloader):
        points, label = data
        points = points.to(dtype=torch.float)
        netpoints = points.transpose(2, 1)
        points = points.view(-1,6)[:, :3]
        model = model.eval()
        pred = model(netpoints)
        pred = pred.view(-1, num_classes)
        pred = torch.max(pred, 1)[1]
        label = label.view(-1)
        
        IOU.append(visualizer.calIOU(pred, label))
        visualizer.MakePCD(points, pred, str(opt.testarea) + '_TEST/' + str(i) + '_' + opt.outn)
        print('saving visualization file ' + str(i) + '_' + opt.outn)
        
print('test data iou is as followed:')

for i, iou in enumerate(IOU):
    print('IOU of room No.%d: %s is %f' % (i, test_dataset.room_names[i], iou))
    