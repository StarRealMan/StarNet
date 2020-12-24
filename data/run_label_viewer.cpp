#include <opencv2/opencv.hpp>

#include <iostream>
#include <fstream>
#include <string>

cv::Vec3b Num2Color(uchar label_num);

int main(int argc, char** argv)
{
    if(argc != 3)
    {
        std::cout << "Need 2 argument for Dataset and Image Num!" << std::endl;

        return 0;
    }

    char image_num[8];
    std::string file_name = "../SUNRGBD/";
    sprintf(image_num, "%06d", std::stoi(argv[2]));
    std::string image_str = image_num;

    if(!strcmp(argv[1],"test"))
    {
        file_name += "test_labels/img13labels-" + image_str + ".png";
    }
    else if(!strcmp(argv[1],"train"))
    {
        file_name += "train_labels/img13labels-" + image_str + ".png";
    }
    else
    {
        file_name = './savings/F_TEST/'+std::stoi(argv[2])+'_test.jpg';
    }
    
    cv::Mat label_pic = cv::imread(file_name, cv::IMREAD_GRAYSCALE);
    cv::Mat show_pic(label_pic.rows, label_pic.cols, CV_8UC3);

    for(int i=0;i<show_pic.rows;i++)  
    {  
        for(int j=0;j<show_pic.cols;j++)  
        {
            cv::Vec3b color;
            color = Num2Color(label_pic.at<uchar>(i,j));
            show_pic.at<cv::Vec3b>(i,j) = color;
        }
    }

    cv::Mat RGB,Depth;

    if(!strcmp(argv[1],"test"))
    {
        RGB = cv::imread("../SUNRGBD/test_dataset/rgb/img-" + image_str + ".jpg");
        image_str = argv[2];
        Depth = cv::imread("../SUNRGBD/test_dataset/depth/" + image_str + ".png");
    }
    else if(!strcmp(argv[1],"train"))
    {
        RGB = cv::imread("../SUNRGBD/train_dataset/rgb/img-" + image_str + ".jpg");
        image_str = argv[2];
        Depth = cv::imread("../SUNRGBD/train_dataset/depth/" + image_str + ".png");
    }

    if(strcmp(argv[1],"eval"))
    {
        cv::imshow("RGB", RGB);
        cv::imshow("Depth",Depth);
    }
    cv::imshow("Label", show_pic);

    cv::waitKey(0);

    return 0;
}

cv::Vec3b Num2Color(uchar label_num)
{
    cv::Vec3b color;

    if(label_num == 1)
        color = cv::Vec3b(127,255,0);
    else if(label_num == 2)
        color = cv::Vec3b(0,127,255);
    else if(label_num == 3)
        color = cv::Vec3b(255,0,0);
    else if(label_num == 4)
        color = cv::Vec3b(255,0,255);
    else if(label_num == 5)
        color = cv::Vec3b(0,255,0);
    else if(label_num == 6)
        color = cv::Vec3b(0,255,255);
    else if(label_num == 7)
        color = cv::Vec3b(255,255,255);
    else if(label_num == 8)
        color = cv::Vec3b(127,255,127);
    else if(label_num == 9)
        color = cv::Vec3b(127,127,255);
    else if(label_num == 10)
        color = cv::Vec3b(127,0,255);
    else if(label_num == 11)
        color = cv::Vec3b(255,255,0);
    else if(label_num == 12)
        color = cv::Vec3b(0,0,255);
    else if(label_num == 13)
        color = cv::Vec3b(255,127,0);
    else
        color = cv::Vec3b(0,0,0);

    return color;
}