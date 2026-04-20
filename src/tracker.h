#pragma once

#include <opencv2/opencv.hpp>
#include <array>
#include <string>

namespace tracker {

struct DetectionResult {
    bool found = false;
    std::array<cv::Point2f, 4> corners{};
    cv::Mat warped;
    cv::Vec3d rvec;
    cv::Vec3d tvec;
    cv::Mat annotated;
};

class TrackerInterface {
public:
    virtual ~TrackerInterface() = default;
    virtual DetectionResult detect(const cv::Mat& input, const std::string& color) const = 0;
};

class QuadContourTracker : public TrackerInterface {
public:
    QuadContourTracker(float originWidth, float originHeight, const cv::Mat& cameraMatrix, int warpSize = 224);
    DetectionResult detect(const cv::Mat& input, const std::string& color) const override;

private:
    cv::Mat buildColorMask(const cv::Mat& hsv, const std::string& color) const;
    static std::array<cv::Point2f, 4> orderCorners(const std::vector<cv::Point>& pts);
    static bool isRectangleLike(const std::array<cv::Point2f, 4>& corners);
    cv::Mat warpToSquare(const cv::Mat& input, const std::array<cv::Point2f, 4>& corners) const;
    std::vector<cv::Point3f> buildObjectPoints() const;
    cv::Mat drawAnnotation(const cv::Mat& input, const std::array<cv::Point2f, 4>& corners, const cv::Vec3d& rvec, const cv::Vec3d& tvec) const;
    static std::string vecToString(const cv::Vec3d& v);

    float originWidth_;
    float originHeight_;
    cv::Mat cameraMatrix_;
    int warpSize_;
};

} // namespace tracker
