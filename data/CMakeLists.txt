cmake_minimum_required(VERSION 2.8)
project(Dataset_Visualizer)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS "-std=c++14 -Wall")
set(CMAKE_CXX_FLAGS_RELEASE  "-std=c++14 -O3 -fopenmp -pthread")

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

############### dependencies ######################

# Eigen
include_directories("/usr/include/eigen3")

# OpenCV
find_package(OpenCV 3.1 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# PCL
find_package(PCL 1.8 REQUIRED COMPONENTS common 
                                         io
                                         features
                                         visualization
                                         filters
                                         keypoints
                                         segmentation)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


set(THIRD_PARTY_LIBS
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
        )

enable_testing()

################### source and driver ######################

file(MAKE_DIRECTORY "./savings")

add_executable(run_dataset_visualizer
               run_dataset_visualizer.cpp)

add_executable(run_label_viewer
               run_label_viewer.cpp)

target_link_libraries(run_dataset_visualizer
                      ${THIRD_PARTY_LIBS})

target_link_libraries(run_label_viewer
                      ${THIRD_PARTY_LIBS})