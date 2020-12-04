#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>

#include <iostream>
#include <fstream>
#include <string>


int main(int argc, char* argv[])
{
    if(argc != 2)
    {
        std::cout << "Need 2 argument for Area and Room!" << std::endl;

        return 0;
    }
    
    std::string Area_num = argv[1];
    Area_num = "Area_" + Area_num;
    std::string Room_num = argv[2];

    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGBA>);
    pcl::visualization::CloudViewer viewer("Dataset Visualizer");
    pcl::PCDWriter Pclwriter;

	std::ifstream read_file;

	read_file.open("../Stanford3dDataset_v1.2_Aligned_Version"
                    + Area_num + "/" + Room_num + "/" + Room_num + ".txt", ios::binary);

    std::string line;
    float pos;
    unsigned char color;
    pcl::PointXYZRGBA point;
    std::vector<pcl::PointXYZRGBA> PointVec;

	while(std::getline(read_file, line))
	{
        std::stringstream lineinput(line);
        lineinput >> pos;
        point.x = pos;
        lineinput >> pos;
        point.y = pos;
        lineinput >> pos;
        point.z = pos;
        lineinput >> color;
        point.r = color;
        lineinput >> color;
        point.g = color;
        lineinput >> color;
        point.b = color;
        PointVec.push_back(point);
	}

    cloud->width = PointVec.size();
    cloud->height = 1;
    cloud->resize(cloud->height * cloud->width);


    for(uint i = 0;i < PointVec.size();i++)
    {
        cloud->points[i] = PointVec[i];
    }

    while(!viewer.wasStopped ())
    {

    }

    Pclwriter.write("../" + Area_num + "_" + Room_num + ".pcd",*(cloud));

    return 0;
}