import numpy as np
import matplotlib.pyplot as plt
from torch import random

class DataPlotter:
    def __init__(self):
        self.xdata = []
        self.ydata = []
        plt.ion()

    def DataloadXY(self, x, y):
        self.xdata.append(x)
        self.ydata.append(y)

    def DataloadY(self, y):
        self.xdata.append(len(self.xdata))
        self.ydata.append(y)
        

    def DataPlot(self):
        plt.clf()
        plt.plot(self.xdata, self.ydata)
        plt.show()


def Type2Color(pred):
    if pred == 0:
        R,G,B = 127,255,0
    elif pred == 1:
        R,G,B = 0,127,255
    elif pred == 2:
        R,G,B = 255,0,0
    elif pred == 3:
        R,G,B = 255,0,255
    elif pred == 4:
        R,G,B = 0,255,0
    elif pred == 5:
        R,G,B = 0,255,255
    elif pred == 6:
        R,G,B = 255,255,255
    elif pred == 7:
        R,G,B = 127,255,127
    elif pred == 8:
        R,G,B = 127,127,255
    elif pred == 9:
        R,G,B = 127,0,255
    elif pred == 10:
        R,G,B = 255,255,0
    elif pred == 11:
        R,G,B = 0,0,255
    elif pred == 12:
        R,G,B = 255,127,0
    elif pred == 13:
        R,G,B = 0,0,0
    else:
        R,G,B = 0,0,0

    color = R<<16 | G<<8 | B | 1<<24
    return color

def MakePCD(points, pred, save_path):
    with open('../data/savings/' + save_path, 'w') as f:
        f.writelines('# .PCD v0.7 - Point Cloud Data file format\n')
        f.writelines('VERSION 0.7\n')
        f.writelines('FIELDS x y z rgb\n')
        f.writelines('SIZE 4 4 4 4\n')
        f.writelines('TYPE F F F U\n')
        f.writelines('COUNT 1 1 1 1\n')
        f.writelines('WIDTH '+str(len(points))+'\n')
        f.writelines('HEIGHT 1\n')
        f.writelines('VIEWPOINT 0 0 0 1 0 0 0\n')
        f.writelines('POINTS '+str(len(points))+'\n')
        f.writelines('DATA ascii\n')
        for i, point in enumerate(points):
            f.writelines(str(point[0].item())+' '+str(point[1].item())+' '+str(point[2].item())+' '+str(Type2Color(pred[i]))+'\n')

def calIOU(pred, label):
    correct_num = 0
    for i, data in enumerate(pred):
        if data == label[i]:
            correct_num = correct_num + 1
    return correct_num/len(pred)


if __name__ == '__main__':
    
    # dataplotter = DataPlotter()
    # for i in range(50):
    #     dataplotter.DataloadY(i**2)
    #     dataplotter.DataPlot()
    #     plt.pause(0.1)
    
    points = np.random.randn(10,3)
    labels = [0,1,2,3,4,5,6,7,8,9]
    MakePCD(points,labels,'test.pcd')