#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>
#include <vector>
#include <iostream>
#include <cmath>

namespace canopy_cam {

void process_images_in_folder_heavy(const std::string& folder_path) {
    namespace fs = std::filesystem;
    
    if (!fs::exists(folder_path) || !fs::is_directory(folder_path)) {
        std::cerr << "Invalid folder: " << folder_path << std::endl;
        return;
    }
    
    const cv::Size resized_size(320, 180);
    const cv::Scalar lower_green(36, 20, 20);
    const cv::Scalar upper_green(145, 255, 255);
    
    // Pre-create structuring elements
    cv::Mat kernel_open = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::Mat kernel_close = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    
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
        
        // Resize to working resolution
        cv::Mat small_bgr;
        cv::resize(bgr, small_bgr, resized_size, 0, 0, cv::INTER_LINEAR);
        
        // Denoise while preserving edges
        cv::Mat denoised;
        cv::bilateralFilter(small_bgr, denoised, 7, 75, 75);
        
        // Convert to HSV
        cv::Mat hsv;
        cv::cvtColor(denoised, hsv, cv::COLOR_BGR2HSV);
        
        // CLAHE on V channel to normalize lighting and enhance contrast
        std::vector<cv::Mat> hsv_channels;
        cv::split(hsv, hsv_channels);
        
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
        clahe->apply(hsv_channels[2], hsv_channels[2]); // Apply to V channel
        
        cv::Mat hsv_eq;
        cv::merge(hsv_channels, hsv_eq);
        
        // Apply green threshold on equalized HSV
        cv::Mat mask;
        cv::inRange(hsv_eq, lower_green, upper_green, mask);
        
        // Suppress low-saturation areas to avoid false positives
        cv::Mat sat_thresh;
        cv::threshold(hsv_channels[1], sat_thresh, 25, 255, cv::THRESH_BINARY);
        cv::bitwise_and(mask, sat_thresh, mask);
        
        // Morphological cleanup: open (remove noise) then close (fill gaps)
        cv::Mat mask_clean;
        cv::morphologyEx(mask, mask_clean, cv::MORPH_OPEN, kernel_open, cv::Point(-1, -1), 1);
        cv::morphologyEx(mask_clean, mask, cv::MORPH_CLOSE, kernel_close, cv::Point(-1, -1), 1);
        
        // Remove small connected components (area filter)
        cv::Mat labels, stats, centroids;
        int num_labels = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8);
        
        if (num_labels > 1) {
            cv::Mat clean = cv::Mat::zeros(mask.size(), mask.type());
            int min_area = std::max(20, static_cast<int>(0.001 * resized_size.area()));
            
            for (int label = 1; label < num_labels; ++label) {
                int area = stats.at<int>(label, cv::CC_STAT_AREA);
                if (area >= min_area) {
                    clean.setTo(255, labels == label);
                }
            }
            mask = clean;
        }
        
        // Percentage calculation at working scale
        int green_pixels = cv::countNonZero(mask);
        double percentage = static_cast<double>(green_pixels) / static_cast<double>(resized_size.area()) * 100.0;
        int rounded = static_cast<int>(std::round(percentage));
        
        // Upscale mask to original resolution for output
        cv::Mat mask_upscaled;
        cv::resize(mask, mask_upscaled, bgr.size(), 0, 0, cv::INTER_NEAREST);
        
        // Save as <stem>_<percentage>.jpeg in heavy output dir
        std::string stem = entry.path().stem().string();
        fs::path out_dir = entry.path().parent_path() / "hsv_segmentation_heavy";
        fs::create_directories(out_dir);
        
        fs::path out_path = out_dir / (stem + "_" + std::to_string(rounded) + ".jpeg");
        
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
    canopy_cam::process_images_in_folder_heavy(argv[1]);
    return 0;
}
