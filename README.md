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

### Tester

## Visualization

## Bug

## Author

![avatar.png](https://github.com/StarRealMan/StarNet/blob/main/images/avatar.png?raw=true)

Student from HITSZ Automatic Control NRS-lab