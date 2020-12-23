import sys

from numpy.testing._private.utils import nulp_diff
sys.path.append("..")
import os
import numpy as np
import torch
import torch.utils.data as data
import cv2
import argparse

from app import visualizer
from app import dataloader


class S3DISDataset(data.Dataset):
    def __init__(self, root_d, pointnum , Area_code = 5, split = 'train'):
        self.pointnum = pointnum
        self.room_names = []
        self.label_names = ['ceiling','floor','wall','beam','column','window','door',\
                            'stairs','table','chair','sofa','bookcase','board','clutter']
        self.label_codes = {}
        self.points = []
        self.label = []
        room_data = []
        room_label = []
        line_data = []

        for label_num, label_name_e in enumerate(self.label_names):
            self.label_codes[label_name_e] = label_num

        if split == 'train':
            Area_range = [1, 2, 3, 4, 5, 6]
            if Area_code != 0:
                Area_range.pop(Area_code-1)

        else:
            Area_range = [Area_code]

        # lodaing area
        for Area_num in Area_range:
            print("loading Area_" + str(Area_num))

            # loading room
            for root,dirs,files in os.walk(root_d+'/Area_'+str(Area_num)):
                root_split = root.split('/')
                if(root_split[-1] == 'Annotations'):
                    room_name = root_split[-2]
                    print("loading Area_" + str(Area_num) +" "+ room_name)
                    self.room_names.append('Area_'+str(Area_num)+" "+room_name)
                    room_data.clear()
                    room_label.clear()

                    # loading annotation
                    for file in files:
                        label_name = file.split('_')[0]
                        if(label_name != 'Icon'):
                            print("loading Area_" + str(Area_num) +" "+ room_name +" "+ file)
                            with open(root_d+'/Area_'+str(Area_num)+'/'+room_name+'/Annotations/'+file, 'r') as f:
                                lines = f.readlines()
                                
                                # lodaing point
                                for line in lines:
                                    line_data = line.split(' ')

                                    # lodaing number
                                    for i in range(6):
                                        line_data[i] = float(line_data[i])
                                    
                                    room_data.append(line_data)
                                    room_label.append(self.label_codes[label_name])
                                    
                    np_room_data = np.array(room_data)
                    np_room_label = np.array(room_label)
                    self.points.append(np_room_data)
                    self.label.append(np_room_label)

    # out put data size : [BatchSize PointNum PointChannel(XYZRGB)]
    def __getitem__(self, index):
        
        points = self.points[index]
        label = self.label[index]
        choice = np.random.choice(range(len(points)), self.pointnum)
        points = points[choice, :]
        label = label[choice]

        return points, label
        
    def __len__(self):
        
        return len(self.room_names)



class SUNRGBDDataset(data.Dataset):
    # I only loaded the train data in the dataset
    def __init__(self, root_d,  split = 'train'):
        self.rgbdata = []
        self.depthdata = []
        self.labels = []

        print('loading '+split+'_dataset rgb image')
        for root,dirs,files in os.walk(root_d+'/'+split+'_dataset/rgb'):
            for file in files:
                rgb_image = cv2.imread(root_d+'/'+split+'_dataset/rgb/'+file, cv2.IMREAD_COLOR)      # shape : 530(H)*730(W)*3(C)
                rgb_image = cv2.resize(rgb_image,(320,240), cv2.INTER_NEAREST)
                rgb_image = rgb_image.transpose((2,0,1))
                self.rgbdata.append(rgb_image)

        print('loading '+split+'_dataset depth image')
        for root,dirs,files in os.walk(root_d+'/'+split+'_dataset/depth'):
            for file in files:
                d_image = cv2.imread(root_d+'/'+split+'_dataset/depth/'+file, cv2.IMREAD_GRAYSCALE) # shape : 530(H)*730(W)
                d_image = cv2.resize(d_image,(320,240), cv2.INTER_NEAREST)
                d_image = d_image.reshape(1,d_image.shape[0],d_image.shape[1])
                self.depthdata.append(d_image)

        print('loading '+split+'_dataset label image')
        for root,dirs,files in os.walk(root_d+'/'+split+'_labels'):
            for file in files:
                label = cv2.imread(root_d+'/'+split+'_labels/'+file, cv2.IMREAD_GRAYSCALE)          # shape : 530(H)*730(W)
                label = cv2.resize(label,(320,240), cv2.INTER_NEAREST)
                label = label.reshape(1,label.shape[0],label.shape[1])
                self.labels.append(label)


    # out put data size : [BatchSize Channel(RGB or Depth) Image_Height Image_Width]
    def __getitem__(self, index):

        rgb = self.rgbdata[index]
        depth = self.depthdata[index]
        label = self.labels[index]

        return rgb, depth, label

    def __len__(self):

        return len(self.labels)


def GenGroundTruth(pointnum, testarea):
    test_dataset = S3DISDataset('../data/Stanford3dDataset_v1.2_Aligned_Version', pointnum, testarea, split = 'test')
    testdataloader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size = 1,\
                                                num_workers=8, drop_last=False)

    if not os.path.exists('../data/savings/'+str(testarea)+'_GT/'):
        os.makedirs('../data/savings/'+str(testarea)+'_GT/')

    for i, data in enumerate(testdataloader):
        points, label = data
        points = points.to(dtype=torch.float)
        points = points.view(-1,6)[:, :3]
        label = label.view(-1)
        
        visualizer.MakePCD(points, label, str(testarea)+'_GT/'+str(i)+'gt.pcd')
        print('saving visualization file in ' + str(testarea)+ '_GT/' + str(i) + '_' + 'gt.pcd')
    

    # for i in range(test_dataset.__len__()):
    #     points, label = test_dataset.__getitem__(i)
    #     points = points[:, :3]
        
    #     visualizer.MakePCD(points, label, str(testarea)+'_GT/'+str(i)+'gt.pcd')
    #     print('saving visualization file in ' + str(testarea)+ '_GT/' + str(i) + '_' + 'gt.pcd')
    

if __name__ == '__main__':

    #  GenGroundTruth(8192, 1)
    
    dataset = SUNRGBDDataset('../data/SUNRGBD', 'train')
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=1,\
                                             num_workers=8, drop_last=True)
    for i, data in enumerate(dataloader):
        rgb,depth,label = data
        print(rgb)
        print(depth)
        print(label)
        break;

    rgb, depth, label = dataset.__getitem__(0)
    print(rgb)
    print(depth)
    print(label)
    print(len(dataset))