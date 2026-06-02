#pragma once

#include <opencv2/opencv.hpp>

#include <array>
#include <string>
#include <vector>

namespace tracker {

struct PlaneTracker2DResult {
    bool found = false;
    std::array<cv::Point2f, 4> corners{};
    cv::Mat mask;
    cv::Mat warped;
    cv::Mat annotated;
    cv::Mat homography;
    cv::Vec3d rvec;
    cv::Vec3d tvec;
    double contourArea = 0.0;
    double reprojectionError = -1.0;
};

class PlaneTracker2D {
public:
    PlaneTracker2D(float originWidth,
                   float originHeight,
                   const cv::Mat& cameraMatrix,
                   const cv::Mat& distCoeffs = cv::Mat(),
                   int warpSize = 224);

    PlaneTracker2DResult detect(const cv::Mat& input, const std::string& color) const;
    PlaneTracker2DResult detectFromCorners(const cv::Mat& input, const std::array<cv::Point2f, 4>& corners) const;

private:
    struct QuadCandidate {
        bool valid = false;
        double area = 0.0;
        std::vector<cv::Point> contour;
        std::array<cv::Point2f, 4> corners{};
    };

    cv::Mat buildColorMask(const cv::Mat& input, const std::string& color) const;
    static void refineMask(cv::Mat& mask);
    QuadCandidate extractLargestQuad(const cv::Mat& mask) const;
    static std::array<cv::Point2f, 4> orderCorners(const std::vector<cv::Point>& polygon);
    static bool isRectangleLike(const std::array<cv::Point2f, 4>& corners);
    cv::Mat warpToSquare(const cv::Mat& input, const std::array<cv::Point2f, 4>& corners, cv::Mat* homography) const;
    std::vector<cv::Point3f> buildObjectPoints() const;
    double computeReprojectionError(const std::vector<cv::Point3f>& objectPoints,
                                    const std::vector<cv::Point2f>& imagePoints,
                                    const cv::Vec3d& rvec,
                                    const cv::Vec3d& tvec) const;
    cv::Mat drawAnnotation(const cv::Mat& input,
                           const std::vector<cv::Point>& contour,
                           const std::array<cv::Point2f, 4>& corners,
                           const cv::Vec3d& rvec,
                           const cv::Vec3d& tvec,
                           double reprojectionError) const;
    static std::string vecToString(const cv::Vec3d& v);

    float originWidth_;
    float originHeight_;
    double maxReprojectionErrorPx_ = 8.0;
    cv::Mat cameraMatrix_;
    cv::Mat distCoeffs_;
    int warpSize_;
};

}  // namespace tracker
