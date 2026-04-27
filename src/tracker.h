#pragma once

#include <opencv2/opencv.hpp>

#include <array>
#include <string>
#include <vector>

namespace tracker {

struct PoseEstimationResult {
    bool found = false;
    std::array<cv::Point2f, 4> faceCorners{};
    std::vector<cv::Point2f> imageCorners2d;
    std::vector<cv::Point3f> objectCorners3d;
    std::vector<cv::Point3f> cameraCorners3d;
    cv::Mat warpedFace;
    cv::Vec3d rvec;
    cv::Vec3d tvec;
    cv::Mat rotationMatrix;
    double reprojectionError = -1.0;
    cv::Mat mask;
    cv::Mat annotated;
};

class TrackerInterface {
public:
    virtual ~TrackerInterface() = default;
    virtual PoseEstimationResult detect(const cv::Mat& input, const std::string& color) const = 0;
};

class CubeTracker3D : public TrackerInterface {
public:
    CubeTracker3D(float originWidth,
                  float originHeight,
                  float originDepth,
                  const cv::Mat& cameraMatrix,
                  const cv::Mat& distCoeffs = cv::Mat(),
                  int warpSize = 224);

    PoseEstimationResult detect(const cv::Mat& input, const std::string& color) const override;

private:
    cv::Mat buildColorMask(const cv::Mat& hsv, const std::string& color) const;
    std::array<cv::Point2f, 4> extractVisibleFace(const cv::Mat& mask) const;
    static std::array<cv::Point2f, 4> orderCorners(const std::vector<cv::Point>& pts);
    static bool isRectangleLike(const std::array<cv::Point2f, 4>& corners);
    cv::Mat warpFaceToSquare(const cv::Mat& input, const std::array<cv::Point2f, 4>& corners) const;

    std::vector<cv::Point3f> buildFrontFaceObjectPoints() const;
    std::vector<cv::Point3f> buildCubeObjectPoints() const;
    std::vector<cv::Point3f> transformToCameraCoordinates(const std::vector<cv::Point3f>& objectPoints,
                                                          const cv::Mat& rotationMatrix,
                                                          const cv::Vec3d& tvec) const;
    std::vector<cv::Point2f> computeVisibleHullPoints(const std::vector<cv::Point2f>& projectedPoints,
                                                      const std::vector<cv::Point3f>& objectPoints,
                                                      std::vector<cv::Point3f>* visibleObjectPoints) const;
    double computeReprojectionError(const std::vector<cv::Point3f>& objectPoints,
                                    const std::vector<cv::Point2f>& imagePoints,
                                    const cv::Vec3d& rvec,
                                    const cv::Vec3d& tvec) const;
    cv::Mat drawAnnotation(const cv::Mat& input,
                           const std::array<cv::Point2f, 4>& faceCorners,
                           const std::vector<cv::Point2f>& projectedCorners,
                           const cv::Vec3d& rvec,
                           const cv::Vec3d& tvec,
                           double reprojectionError) const;
    static std::string vecToString(const cv::Vec3d& v);

    float originWidth_;
    float originHeight_;
    float originDepth_;
    cv::Mat cameraMatrix_;
    cv::Mat distCoeffs_;
    int warpSize_;
};

}  // namespace tracker
