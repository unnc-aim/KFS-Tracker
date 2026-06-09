#pragma once

#include <opencv2/opencv.hpp>

#include <memory>
#include <string>
#include <vector>

namespace tracker {

struct HeatmapTrackingResult {
    bool valid = false;
    std::vector<cv::Point2f> corners;   // 4 个角点，原图像素坐标
    cv::Mat annotated;                  // 标注可视化图
};

class HeatmapTrackingClassifier {
public:
    explicit HeatmapTrackingClassifier(const std::string& modelPath, int inputSize = 300);
    HeatmapTrackingResult infer(const cv::Mat& image) const;

private:
    int inputSize_ = 300;
    class Impl;
    std::shared_ptr<Impl> impl_;
};

}  // namespace tracker
