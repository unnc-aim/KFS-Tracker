#pragma once

#include <opencv2/opencv.hpp>

#include <array>
#include <memory>
#include <string>

namespace tracker {

struct HeatmapInferResult {
    bool valid = false;
    std::array<cv::Point2f, 4> corners{};
    cv::Mat annotated;
};

struct ClassifierInferResult {
    bool valid = false;
    int classId = -1;
    float classScore = 0.0f;
    std::string className;
};

class HeatmapCornerInfer {
public:
    explicit HeatmapCornerInfer(const std::string& modelPath, int inputSize = 300);
    HeatmapInferResult infer(const cv::Mat& image) const;

private:
    int inputSize_ = 300;
    class Impl;
    std::shared_ptr<Impl> impl_;
};

class ImageClassifierInfer {
public:
    explicit ImageClassifierInfer(const std::string& modelPath, int inputSize = 300);
    ClassifierInferResult infer(const cv::Mat& image) const;

private:
    int inputSize_ = 300;
    class Impl;
    std::shared_ptr<Impl> impl_;
};

}  // namespace tracker

