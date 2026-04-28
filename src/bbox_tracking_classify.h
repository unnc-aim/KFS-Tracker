#pragma once

#include <opencv2/opencv.hpp>

#include <memory>
#include <string>
#include <vector>

namespace tracker {

struct BboxClassificationResult {
    bool valid = false;
    cv::Rect bbox;
    std::vector<cv::Point> bboxContour;
    int classId = -1;
    std::string className;
    float classScore = 0.0F;
    cv::Mat annotated;
};

class BboxTrackingClassifier {
public:
    explicit BboxTrackingClassifier(const std::string& modelPath, int inputSize = 300);
    BboxClassificationResult infer(const cv::Mat& image) const;

private:
    int inputSize_ = 300;
    class Impl;
    std::shared_ptr<Impl> impl_;
};

}  // namespace tracker
