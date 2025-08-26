cmake_minimum_required(VERSION 3.16)
project(canopy_cam_model_testing)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# HSV Segmentation
add_executable(hsv_segmentation thresholding/hsv_segmentation.cpp)
target_link_libraries(hsv_segmentation ${OpenCV_LIBS})

# Heavy HSV Segmentation
add_executable(hsv_segmentation_heavy thresholding/hsv_segmentation_heavy.cpp)
target_link_libraries(hsv_segmentation_heavy ${OpenCV_LIBS})

# LAB Segmentation
add_executable(lab_segmentation thresholding/lab_segmentation.cpp)
target_link_libraries(lab_segmentation ${OpenCV_LIBS})

# K-means Clustering
add_executable(k_means_clustering unsupervised_ml/k_means_clustering.cpp)
target_link_libraries(k_means_clustering ${OpenCV_LIBS})

# Set output directory
set_target_properties(hsv_segmentation hsv_segmentation_heavy lab_segmentation k_means_clustering
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
