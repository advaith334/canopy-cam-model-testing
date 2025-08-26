#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>
#include <vector>
#include <iostream>
#include <cmath>

namespace canopy_cam {

void process_images_in_folder_lab(const std::string& folder_path) {
    namespace fs = std::filesystem;
    
    if (!fs::exists(folder_path) || !fs::is_directory(folder_path)) {
        std::cerr << "Invalid folder: " << folder_path << std::endl;
        return 1;
    }
    
    const cv::Size resized_size(320, 180);
    const int a_max = 135;
    const int b_min = 110;
    
    // Output directory: <folder>/lab_segmentation
    fs::path out_dir = fs::path(folder_path) / "lab_segmentation";
    fs::create_directories(out_dir);
    
    std::vector<fs::path> image_paths;
    for (const fs::directory_entry& entry : fs::directory_iterator(folder_path)) {
        if (!entry.is_regular_file()) continue;
        
        std::string ext = entry.path().extension().string();
        if (ext != ".jpg" && ext != ".jpeg" && ext != ".JPG" && ext != ".JPEG") continue;
        
        image_paths.push_back(entry.path());
    }
    
    cv::Mat kernel_open = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::Mat kernel_close = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    
    for (const fs::path& img_path : image_paths) {
        cv::Mat bgr = cv::imread(img_path.string(), cv::IMREAD_COLOR);
        if (bgr.empty()) {
            std::cerr << "Failed to read image: " << img_path.string() << std::endl;
            continue;
        }
        
        // Resize and mild denoise
        cv::Mat small_bgr;
        cv::resize(bgr, small_bgr, resized_size, 0, 0, cv::INTER_LINEAR);
        cv::Mat denoised;
        cv::bilateralFilter(small_bgr, denoised, 5, 60, 60);
        
        // Convert to LAB
        cv::Mat lab;
        cv::cvtColor(denoised, lab, cv::COLOR_BGR2LAB);
        std::vector<cv::Mat> lab_channels;
        cv::split(lab, lab_channels);
        
        // Apply CLAHE to L channel
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
        clahe->apply(lab_channels[0], lab_channels[0]);
        
        // Threshold: A low (<= a_max), B high (>= b_min)
        cv::Mat a_mask, b_mask;
        cv::threshold(lab_channels[1], a_mask, a_max, 255, cv::THRESH_BINARY_INV);
        cv::threshold(lab_channels[2], b_mask, b_min, 255, cv::THRESH_BINARY);
        
        cv::Mat mask;
        cv::bitwise_and(a_mask, b_mask, mask);
        
        // Optional: limit to reasonable brightness to avoid glare/sky
        // Keep mid-to-high luminance but drop near-white saturation: 20..230
        cv::Mat L_clip;
        cv::inRange(lab_channels[0], 20, 230, L_clip);
        cv::bitwise_and(mask, L_clip, mask);
        
        // Morphological refine
        cv::Mat mask_clean;
        cv::morphologyEx(mask, mask_clean, cv::MORPH_OPEN, kernel_open, cv::Point(-1, -1), 1);
        cv::morphologyEx(mask_clean, mask, cv::MORPH_CLOSE, kernel_close, cv::Point(-1, -1), 1);
        
        // Remove tiny blobs
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
        
        // Percentage at working scale
        int green_pixels = cv::countNonZero(mask);
        double percentage = static_cast<double>(green_pixels) / static_cast<double>(resized_size.area()) * 100.0;
        int rounded = static_cast<int>(std::round(percentage));
        
        // Save upscaled mask
        cv::Mat mask_upscaled;
        cv::resize(mask, mask_upscaled, bgr.size(), 0, 0, cv::INTER_NEAREST);
        
        std::string stem = img_path.stem().string();
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
    canopy_cam::process_images_in_folder_lab(argv[1]);
    return 0;
}
