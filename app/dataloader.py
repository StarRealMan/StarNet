import os
import numpy as np
import torch
import torch.utils.data as data
import cv2


class S3DISDataset(data.Dataset):
    def __init__(self,root_d):
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

        for Area_num in range(1):
            print("loading Area_" + str(Area_num+1))
            for root,dirs,files in os.walk(root_d+'/Area_'+str(Area_num+1)):
                root_split = root.split('/')
                if(root_split[-1] == 'Annotations'):
                    room_name = root_split[-2]
                    print("loading Area_" + str(Area_num+1) +" "+ room_name)
                    self.room_names.append('Area_'+str(Area_num+1)+room_name)
                    
                    room_data.clear()
                    room_label.clear()
                    for file in files:
                        label_name = file.split('_')[0]
                        if(label_name != 'Icon'):
                            print("loading Area_" + str(Area_num+1) +" "+ room_name +" "+ file)
                            with open(root_d+'/Area_'+str(Area_num+1)+'/'+room_name+'/Annotations/'+file, 'r') as f:                                
                                lines = f.readlines()
                                for line in lines:
                                    line_data = line.split(' ')
                                    for i in range(6):
                                        line_data[i] = float(line_data[i])
                                    
                                    room_data.append(line_data)
                                    room_label.append(self.label_codes[label_name])

                            self.points.append(room_data)
                            self.label.append(room_label)

    # out put data size : [BatchSize PointChannel(XYZRGB) PointNum]
    def __getitem__(self, index):
        
        points = np.array(self.points[index])
        label = np.array(self.label[index])

        return points, label
        
    def __len__(self):
        
        return len(self.room_names)



class SUNRGBDDataset(data.Dataset):
    # I only loaded the train data in the dataset
    def __init__(self,root_d):
        self.rgbdata = []
        self.depthdata = []
        self.labels = []
        for root,dirs,files in os.walk(root_d+'/train_dataset/rgb'):
            for file in files:
                rgb_image = cv2.imread(root_d+'/train_dataset/rgb/'+file, cv2.IMREAD_COLOR)      # shape : 530(H)*730(W)*3(C)
                rgb_image = cv2.resize(rgb_image,(320,240), cv2.INTER_NEAREST)
                rgb_image = rgb_image.transpose((2,0,1))
                self.rgbdata.append(rgb_image)

        for root,dirs,files in os.walk(root_d+'/train_dataset/depth'):
            for file in files:
                d_image = cv2.imread(root_d+'/train_dataset/depth/'+file, cv2.IMREAD_GRAYSCALE)      # shape : 530(H)*730(W)
                d_image = cv2.resize(d_image,(320,240), cv2.INTER_NEAREST)
                d_image = d_image.reshape(1,d_image.shape[0],d_image.shape[1])
                self.depthdata.append(d_image)

        for root,dirs,files in os.walk(root_d+'/train_labels'):
            for file in files:
                label = cv2.imread(root_d+'/train_labels/'+file, cv2.IMREAD_GRAYSCALE)      # shape : 530(H)*730(W)
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

if __name__ == '__main__':
    # dataset = S3DISDataset('../data/Stanford3dDataset_v1.2_Aligned_Version')
    # dataset = S3DISDataset('../data/test_dataset')
    # dataloader = torch.utils.data.DataLoader(dataset,shuffle=True,batch_size=1,num_workers=4)

    # for i, data in enumerate(dataloader):
    #     points,label = data
    #     print(points)
    #     print(label)
    #     break;

    dataset = SUNRGBDDataset('../data/SUNRGBD')
    dataloader = torch.utils.data.DataLoader(dataset,shuffle=True,batch_size=1,num_workers=4)
    for i, data in enumerate(dataloader):
        rgb,depth,label = data
        print(rgb)
        print(depth)
        print(label)
        break;


    print(dataset.__getitem__(0))
    print(dataset.__len__())