#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>
#include <vector>
#include <iostream>
#include <cmath>

namespace canopy_cam {

void process_images_in_folder(const std::string& folder_path) {
    namespace fs = std::filesystem;
    
    if (!fs::exists(folder_path) || !fs::is_directory(folder_path)) {
        std::cerr << "Invalid folder: " << folder_path << std::endl;
        return;
    }
    
    const cv::Size resized_size(320, 180);
    const cv::Scalar lower_green(36, 20, 20);
    const cv::Scalar upper_green(145, 255, 255);
    
    for (const fs::directory_entry& entry : fs::directory_iterator(folder_path)) {
        if (!entry.is_regular_file()) continue;
        
        std::string ext = entry.path().extension().string();
        if (ext != ".jpg" && ext != ".jpeg" && ext != ".JPG" && ext != ".JPEG") continue;
        
        std::string input_path = entry.path().string();
        cv::Mat bgr = cv::imread(input_path, cv::IMREAD_COLOR);
        if (bgr.empty()) {
            std::cerr << "Failed to read image: " << input_path << std::endl;
            continue;
        }
        
        // Resize for lightweight processing
        cv::Mat small_bgr;
        cv::resize(bgr, small_bgr, resized_size, 0, 0, cv::INTER_LINEAR);
        
        // Convert to HSV and threshold green
        cv::Mat hsv;
        cv::cvtColor(small_bgr, hsv, cv::COLOR_BGR2HSV);
        cv::Mat mask;
        cv::inRange(hsv, lower_green, upper_green, mask);
        
        int green_pixels = cv::countNonZero(mask);
        double percentage = static_cast<double>(green_pixels) / static_cast<double>(resized_size.area()) * 100.0;
        int rounded = static_cast<int>(std::round(percentage));
        
        // Upscale mask to original size for output visualization
        cv::Mat mask_upscaled;
        cv::resize(mask, mask_upscaled, bgr.size(), 0, 0, cv::INTER_NEAREST);
        
        // Compose output filename: <stem>_<percentage>.jpeg in the same directory
        std::string stem = entry.path().stem().string();
        fs::path out_path = entry.path().parent_path() / (stem + "_" + std::to_string(rounded) + ".jpeg");
        
        if (!cv::imwrite(out_path.string(), mask_upscaled)) {
            std::cerr << "Failed to write: " << out_path.string() << std::endl;
        }
    }
}

} // namespace canopy_cam

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <folder>" << std::endl;
        return 1;
    }
    canopy_cam::process_images_in_folder(argv[1]);
    return 0;
}
