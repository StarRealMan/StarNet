import os
import sys
import cv2
import numpy as np
import torch
import torch.utils.data as data
from tqdm import tqdm


class S3DISDataset(data.Dataset):
    def __init__(self,root_d):
        self.room_names = []
        self.label_names = ['ceiling','floor','wall','beam','column','window','door','table','chair','sofa','bookcase','board','clutter']
        self.label_codes = {}
        self.data = []
        self.label = []

        for label_num, label_name_e in enumerate(self.label_names):
            self.label_codes[label_name_e] = label_num

        for Area_num in range(6):
            for root,dirs,files in os.walk(root_d+'/Area_'+str(Area_num+1)):
                root_split = root.split('/')
                if(root_split[-1] == 'Annotations'):
                    room_name = root_split[-2]
                    self.room_names.append('Area_'+str(Area_num+1)+room_name)
                    
                    room_data = []
                    room_label = []
                    for file in files:
                        label_name = file.split('_')[0]
                        if(label_name != 'Icon'):

                            with open(root_d+'/Area_'+str(Area_num+1)+'/'+room_name+'/Annotations/'+file, 'r') as f:                                
                                lines = f.readlines()
                                for line in lines:
                                    room_data.append(line.split(' '))
                                    room_label.append(self.label_codes[label_name])

                            self.data.append(room_data)
                            self.label.append(room_label)

        
    def __getitem__(self, index):
        self.data[index]
        self.label[index]

        return , 
        
    def __len__(self):
        
        return self.room_names.size()

if __name__ == '__main__':
    dataset = S3DISDataset('../data/Stanford3dDataset_v1.2_Aligned_Version')
