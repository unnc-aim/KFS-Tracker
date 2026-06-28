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
    float minConfidence = 0.0f;         // 4 个角点热力图峰值的最小值（0~1）
    float aspectRatio = 0.0f;           // 四边形最大边长 / 最小边长（几何合理性）
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
