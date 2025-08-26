#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>

namespace canopy_cam {

void process_images_in_folder_kmeans(const std::string& folder_path) {
    namespace fs = std::filesystem;
    
    if (!fs::exists(folder_path) || !fs::is_directory(folder_path)) {
        std::cerr << "Invalid folder: " << folder_path << std::endl;
        return;
    }
    
    const cv::Size resized_size(320, 180);
    
    // Output directory: <folder>/k_means_clustering
    fs::path out_dir = fs::path(folder_path) / "k_means_clustering";
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
        
        // K-means clustering (k=2) on LAB features
        cv::Mat lab_small = lab.clone();
        lab_small.convertTo(lab_small, CV_32F, 1.0/255.0);
        
        int total_pixels = resized_size.width * resized_size.height;
        cv::Mat samples = lab_small.reshape(1, total_pixels);
        
        cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 50, 1e-3);
        const int K = 2;
        
        cv::Mat labels, centers;
        double compactness = cv::kmeans(samples, K, labels, criteria, 10, cv::KMEANS_PP_CENTERS, centers);
        
        // Determine which cluster represents plants using a composite heuristic
        // Compute per-cluster means in LAB and HSV spaces and spatial prior
        int h = resized_size.height;
        int w = resized_size.width;
        
        // Prepare HSV for statistics
        cv::Mat hsv_for_stats;
        cv::cvtColor(denoised, hsv_for_stats, cv::COLOR_BGR2HSV);
        std::vector<cv::Mat> hsv_channels;
        cv::split(hsv_for_stats, hsv_channels);
        
        // Build coordinate grid for spatial prior (favor lower 2/3 of image)
        cv::Mat yy = cv::Mat::zeros(h, w, CV_32S);
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                yy.at<int>(y, x) = y;
            }
        }
        
        std::vector<double> scores(K);
        for (int c = 0; c < K; ++c) {
            cv::Mat mask_c = (labels == c).reshape(1, h);
            int count = cv::countNonZero(mask_c);
            
            if (count == 0) {
                scores[c] = -1e9;
                continue;
            }
            
            // LAB stats (from normalized centers)
            double a_mean = centers.at<float>(c, 1);  // 0..1
            double b_mean = centers.at<float>(c, 2);  // 0..1
            
            // HSV stats (compute from pixels)
            double H_mean = 0.0, S_mean = 0.0, V_mean = 0.0;
            int valid_count = 0;
            
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    if (mask_c.at<uchar>(y, x)) {
                        H_mean += hsv_channels[0].at<uchar>(y, x);
                        S_mean += hsv_channels[1].at<uchar>(y, x);
                        V_mean += hsv_channels[2].at<uchar>(y, x);
                        valid_count++;
                    }
                }
            }
            
            if (valid_count > 0) {
                H_mean /= valid_count;
                S_mean /= valid_count;
                V_mean /= valid_count;
            }
            
            // Normalize HSV ranges to 0..1
            double Hn = H_mean / 179.0;
            double Sn = S_mean / 255.0;
            double Vn = V_mean / 255.0;
            
            // Hue closeness to green band (~35..85 on 0..179); compute proximity score
            double green_low = 35.0/179.0, green_high = 85.0/179.0;
            double hue_dist;
            if (Hn < green_low) {
                hue_dist = green_low - Hn;
            } else if (Hn > green_high) {
                hue_dist = Hn - green_high;
            } else {
                hue_dist = 0.0;
            }
            double hue_score = 1.0 - std::min(1.0, hue_dist / (50.0/179.0));  // within band => ~1
            
            // Spatial prior: fraction of pixels in lower 2/3 of the image
            int lower_region_count = 0;
            for (int y = h/3; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    if (mask_c.at<uchar>(y, x)) {
                        lower_region_count++;
                    }
                }
            }
            double spatial_frac = static_cast<double>(lower_region_count) / static_cast<double>(count);
            
            // Composite score: lower A, higher S, good green hue, moderate V, more in lower region
            scores[c] = (-a_mean) * 1.0 + (Sn) * 0.8 + (hue_score) * 1.0 + 
                        ((1.0 - Vn)) * 0.3 + (spatial_frac) * 0.5 + (b_mean) * 0.2;
        }
        
        int plant_cluster = std::max_element(scores.begin(), scores.end()) - scores.begin();
        
        // Create mask from plant cluster
        cv::Mat mask = (labels == plant_cluster).reshape(1, h);
        mask.convertTo(mask, CV_8U, 255);
        
        // Optional: clip by luminance to avoid sky/glare
        cv::Mat L_clip;
        cv::inRange(lab_channels[0], 20, 240, L_clip);
        cv::bitwise_and(mask, L_clip, mask);
        
        // Morphological refine
        cv::Mat mask_clean;
        cv::morphologyEx(mask, mask_clean, cv::MORPH_OPEN, kernel_open, cv::Point(-1, -1), 1);
        cv::morphologyEx(mask_clean, mask, cv::MORPH_CLOSE, kernel_close, cv::Point(-1, -1), 1);
        
        // Remove tiny blobs
        cv::Mat labels_cc, stats_cc, centroids_cc;
        int num_labels = cv::connectedComponentsWithStats(mask, labels_cc, stats_cc, centroids_cc, 8);
        
        if (num_labels > 1) {
            cv::Mat clean = cv::Mat::zeros(mask.size(), mask.type());
            int min_area = std::max(20, static_cast<int>(0.001 * resized_size.area()));
            
            for (int label = 1; label < num_labels; ++label) {
                int area = stats_cc.at<int>(label, cv::CC_STAT_AREA);
                if (area >= min_area) {
                    clean.setTo(255, labels_cc == label);
                }
            }
            mask = clean;
        }
        
        // Calculate percentage at working scale
        int plant_pixels = cv::countNonZero(mask);
        double percentage = static_cast<double>(plant_pixels) / static_cast<double>(resized_size.area()) * 100.0;
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
    canopy_cam::process_images_in_folder_kmeans(argv[1]);
    return 0;
}
