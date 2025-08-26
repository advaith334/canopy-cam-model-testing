#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/float32.hpp>

namespace canopy_cam
{

class CanopyCoverage : public rclcpp::Node
{
public:
  explicit CanopyCoverage(const rclcpp::NodeOptions & options) : Node("canopy_coverage", options)
  {
    this->declare_parameter("in", "/canopy_cam/image_raw_throttle");
    std::string input_topic = this->get_parameter("in").as_string();

    canopy_coverage_publisher_ =
      this->create_publisher<std_msgs::msg::Float32>("canopy_coverage", 10);

    using std::placeholders::_1;
    image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      input_topic, 10, std::bind(&CanopyCoverage::process_frame, this, _1));

    // Preallocate small Mats for reuse
    resized_size_ = cv::Size(320, 180);  // ~1/16 of 720p
    hsv_.create(resized_size_, CV_8UC3);
    mask_.create(resized_size_, CV_8UC1);
  }

private:
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr canopy_coverage_publisher_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;

  cv::Size resized_size_;
  cv::Mat hsv_, mask_;

  void process_frame(const sensor_msgs::msg::Image::ConstSharedPtr msg)
  {
    if (msg->encoding != "yuv422_yuy2") {
      static int warn_count = 0;
      if (warn_count++ < 5)
        RCLCPP_WARN(this->get_logger(), "Expected 'yuv422_yuy2', got '%s'", msg->encoding.c_str());
      return;
    }

    const int width = msg->width;
    const int height = msg->height;
    if (width == 0 || height == 0) return;

    // Convert to YUYV image
    cv::Mat yuyv(height, width, CV_8UC2, const_cast<unsigned char *>(msg->data.data()), msg->step);

    // Convert YUYV to BGR
    cv::Mat bgr;
    cv::cvtColor(yuyv, bgr, cv::COLOR_YUV2BGR_YUY2);

    // Resize for lightweight processing
    cv::Mat small_bgr;
    cv::resize(bgr, small_bgr, resized_size_, 0, 0, cv::INTER_LINEAR);

    // Convert to HSV
    cv::cvtColor(small_bgr, hsv_, cv::COLOR_BGR2HSV);

    // Green mask
    const cv::Scalar lower_green(36, 20, 20);
    const cv::Scalar upper_green(145, 255, 255);
    cv::inRange(hsv_, lower_green, upper_green, mask_);

    int green_pixels = cv::countNonZero(mask_);
    double percentage = static_cast<double>(green_pixels) / (resized_size_.area()) * 100.0;

    // Publish canopy coverage
    std_msgs::msg::Float32 out;
    out.data = static_cast<float>(percentage);
    canopy_coverage_publisher_->publish(out);
  }
};

}  // namespace canopy_cam

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(canopy_cam::CanopyCoverage)
