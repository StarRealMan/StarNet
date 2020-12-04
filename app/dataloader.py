import os
import sys
import cv2
import numpy as np
import torch
import torch.utils.data as data
from tqdm import tqdm


class ImageNetDataset(data.Dataset):
    def __init__(self,root,type='train'):
        labels_t = []
        image_names = []                                                    #root = ../dataset/tiny-imagenet-200/
        with open(root+'wnids.txt') as wnid:                                #open ../dataset/tiny-imagenet-200/wnids.txt
            for line in wnid:
                labels_t.append(line.strip('\n'))                           #label_t 200 labels(strip removes \n)
        for label in labels_t:
            txt_path = root+'train/'+label+'/'+label+'_boxes.txt'           #txt path = root/'label'/'label'_boxes.txt (info of images in this label)
            image_name = []
            with open(txt_path) as txt:
                for line in txt:                                            #line in txt store the info of 500 images belongs to this label
                    image_name.append(line.strip('\n').split('\t')[0])      #image_name = [500*name_JPEG...]
            image_names.append(image_name)                                  #image_names = [200*image_name...]
                                                                            #image_names and labels_t have the same order
        val_labels_t = []
        self.val_labels = []
        val_names = []
        with open(root+'val/val_annotations.txt') as txt:                   #validation same way
            for line in txt:
                val_names.append(line.strip('\n').split('\t')[0])           #validation image name
                val_labels_t.append(line.strip('\n').split('\t')[1])        #validation label
        for i in range(len(val_labels_t)):
            for i_t in range(len(labels_t)):
                if val_labels_t[i] == labels_t[i_t]:                        
                    self.val_labels.append(i_t)                             #val_labels stores the code_num of labels in val
        self.val_labels = np.array(self.val_labels)                         #change dtype to numpy array
        self.type = type
        if type == 'train':
            self.images = []
            for i,label in enumerate(labels_t):                             #traverse the labels
                image = []
                for image_name in image_names[i]:                           #find all the images in a label
                    image_path = os.path.join(root+'train', label, 'images', image_name) 
                    image.append(cv2.imread(image_path))                    #image = [500*numpy_mat...]
                self.images.append(image)                                   #images = [200*image...]    
            self.images = np.array(self.images)                             #images = [200,500,64,64,3] [label image height width channel]
            self.images = self.images.reshape(-1, 64, 64, 3)                #reshape to [batch(full num) height width channel]
        elif type == 'val':
            self.val_images = []
            for val_image in val_names:                                     #validation same way
                val_image_path = os.path.join(root+'val/images', val_image)
                self.val_images.append(cv2.imread(val_image_path))
            self.val_images = np.array(self.val_images)
        
    def __getitem__(self, index):
        label = []
        image = []
        if self.type == 'train':
            label = index//500                                              #python//运算符表示做除法之后对商向下取整，由image的index得到label（500一组连续排列）
            image = self.images[index].transpose(2,0,1)                     #transpose [64 64 3] -> [3 64 64]
        if self.type == 'val':
            label = self.val_labels[index]
            image = self.val_images[index].transpose(2,0,1)
        return label, image
        
    def __len__(self):
        len = 0
        if self.type == 'train':
            len = self.images.shape[0]
        if self.type == 'val':
            len = self.val_images.shape[0]
        return len