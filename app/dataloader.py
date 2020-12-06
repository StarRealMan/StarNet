import os
import sys
import cv2
import numpy as np
import torch
import torch.utils.data as data
from tqdm import tqdm


class S3DISDataset(data.Dataset):
    def __init__(self,root_d):

        for Area_num in range(6):
            for root,dirs,files in os.walk(root_d):
                print('root:',root)
                print('dirs:',dirs)
                print('files:',files)
                


                
        # with open(root+'Area_2') as Farea2:
        #     for line in Farea2:

        # with open(root+'Area_3') as Farea3:
        #     for line in Farea3:

        # with open(root+'Area_4') as Farea4:
        #     for line in Farea4:

        # with open(root+'Area_5') as Farea5:
        #     for line in Farea5:

        # with open(root+'Area_6') as Farea6:
        #     for line in Farea6:
        
    def __getitem__(self, index):
        
        return label, image
        
    def __len__(self):
        
        return len

if __name__ == '__main__':
    dataset = S3DISDataset('../data/Stanford3dDataset_v1.2_Aligned_Version/Area_1')
