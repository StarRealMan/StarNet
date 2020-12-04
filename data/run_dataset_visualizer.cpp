#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/transforms.h>

#include <iostream>
#include <fstream>
#include <string>


int main(int argc, char* argv[])
{
    if(argc != 3)
    {
        std::cout << "Need 2 argument for Area and Room!" << std::endl;

        return 0;
    }
    
    std::string Area_num = argv[1];
    std::string Area_name = "Area_" + Area_num;
    std::string Room_name = argv[2];

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PCDWriter Pclwriter;

	std::ifstream read_file;

    std::cout << "Opening : ../Stanford3dDataset_v1.2_Aligned_Version/"
                    + Area_name + "/" + Room_name + "/" + Room_name + ".txt" << std::endl;

	read_file.open("../Stanford3dDataset_v1.2_Aligned_Version/"
                    + Area_name + "/" + Room_name + "/" + Room_name + ".txt", ios::binary);

    std::string line;
    float pos;
    unsigned int color;
    pcl::PointXYZRGB point;
    float midpoint[3] = {0};
    std::vector<pcl::PointXYZRGB> PointVec;

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

        midpoint[0] -= point.x;
        midpoint[1] -= point.y;
        midpoint[2] -= point.z;
	}    

    cloud->width = PointVec.size();
    cloud->height = 1;
    cloud->resize(cloud->height * cloud->width);

    for(uint i = 0;i < PointVec.size();i++)
    {
        cloud->points[i] = PointVec[i];
    }

    std::cout << "File read successfully" << std::endl;

    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    Eigen::Vector3f translation;
    translation << midpoint[0]/PointVec.size(), midpoint[1]/PointVec.size(), midpoint[2]/PointVec.size();
    transform.translate(translation);
    pcl::transformPointCloud(*cloud, *cloud, transform);

    pcl::visualization::CloudViewer viewer("Dataset Visualizer");
    viewer.showCloud(cloud);

    while(!viewer.wasStopped ())
    {

    }

    // Pclwriter.write("../savings/" + Area_num + "_" + Room_num + ".pcd",*(cloud));

    return 0;
}