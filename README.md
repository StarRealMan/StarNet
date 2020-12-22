# StarNet
Graduation Project: A deep neural network for point cloud semantic segmentation, part of the SSVIO project

Dataset:

@InProceedings

armeni_cvpr16,

title     : 3D Semantic Parsing of Large-Scale Indoor Spaces,

author    : Iro Armeni and Ozan Sener and Amir R. Zamir and Helen Jiang and Ioannis Brilakis and Martin Fischer and Silvio Savarese,

booktitle :Proceedings of the IEEE International Conference on Computer Vision and Pattern Recognition,

year      : 2016

## REMEMBER!
Do not try use point num more than 4096 or batch size more than 32 to train sceneseg  
if your GPU memory is less than 12 GB 

## Requirement
* Pytorch:<https://pytorch.org/>
* argparse:
```
pip install argparse
```
* tqdm:
```
pip install tqdm
```
* PCL:<https://pointclouds.org/>
* Stanford3dDataset:<https://console.cloud.google.com/storage/browser/dataset_cvpr16;tab=objects?pli=1&prefix&forceOnObjectsSortingFiltering=false>

## Usage

* Download Stanford3dDataset to {YOUR_DIRECTORY}/StarNet/data
* Name the directory as Stanford3dDataset_v1.2_Aligned_Version
* Download SUNRGBD data **For Semantic Segmentation** to {YOUR_DIRECTORY}/StarNet/data
* Name the directory as SUNRGBD

### Dataset Visualizer
* Go to {YOUR_DIRECTORY}/StarNet/data
* Run following code
```
mkdir build
cd ./build
cmake ..
make
```
* After generating bin file, go to {YOUR_DIRECTORY}/SSVIO/bin
* Run following code to show the room
```
./run_data_visualizer {Area_num} {Room_name}
```
* Run following code to show rgbd image and semantic segmentation
```
./run_label_viewer {test/train} {image_num}
```
* Run this to remove saved pcd data:
```
bash ./removedata.sh
```

### Trainer
* Run the following code at /app/ to train your model
```
python sceneseg_train.py [optins]
```
option include:
--batchsize for input batch size  
--pointnum for points per room/sample  
--nepoch for number of epochs to train for  
--dataset for dataset path  
--outn for output model name  
--model for history model path  
--workers for number of workers to load data

### Tester
* Run the following code at /app/ to test the model you trained
```
python sceneseg_test.py [optins]
```
option include:
--dataset for dataset path  
--model for history model path  
--pointnum for points per room/sample  
--outn for output file name  
--workers for number of workers to load data

## Visualization
* While Training, it will generate loss chart at the same time.
* Using sceneseg_test.py, just type in the room/sample num will save the result in pcd format at /data/savings 

## Bug
Error when using test app. It predict all the pointto be 'sofa'(class 11)  
Using model trained Dec.22, 300 epoch with all Area.

## Author

![avatar.png](https://github.com/StarRealMan/StarNet/blob/main/images/avatar.png?raw=true)

Student from HITSZ Automatic Control NRS-lab