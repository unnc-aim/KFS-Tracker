#pragma once

#include <opencv2/opencv.hpp>

#include <array>
#include <limits>
#include <string>
#include <vector>

namespace tracker {

struct PoseEstimationResult {
    bool found = false;
    std::array<cv::Point2f, 6> hexagonCorners{};
    std::vector<cv::Point2f> imageCorners2d;
    std::vector<cv::Point3f> objectCorners3d;
    std::vector<cv::Point3f> cameraCorners3d;
    std::vector<cv::Point2f> projectedCubeCorners2d;
    cv::Mat mask;
    cv::Mat annotated;
    cv::Vec3d rvec;
    cv::Vec3d tvec;
    cv::Mat rotationMatrix;
    double reprojectionError = -1.0;
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
    struct HexagonCandidate {
        bool valid = false;
        double area = 0.0;
        std::vector<cv::Point> contour;
        std::array<cv::Point2f, 6> corners{};
    };

    struct EdgeInfo {
        int startIndex = 0;
        int endIndex = 0;
        int cluster = -1;
        float length = 0.0f;
        float angle = 0.0f;
        cv::Point2f direction;
        cv::Vec3f line;
    };

    struct PoseCandidate {
        bool valid = false;
        std::array<cv::Point2f, 6> imageCorners{};
        std::vector<cv::Point3f> objectCorners;
        std::vector<cv::Point2f> projectedCorners;
        cv::Vec3d rvec;
        cv::Vec3d tvec;
        double reprojectionError = std::numeric_limits<double>::infinity();
        double totalScore = std::numeric_limits<double>::infinity();
    };

    cv::Mat buildColorMask(const cv::Mat& input, const std::string& color) const;
    static cv::Mat applyClaheToValueChannel(const cv::Mat& input);
    static void fillMaskHoles(cv::Mat& mask);
    HexagonCandidate extractLargestHexagon(const cv::Mat& mask) const;
    static std::array<cv::Point2f, 6> orderPolygonCorners(const std::vector<cv::Point2f>& points);
    static std::vector<EdgeInfo> buildEdgeInfos(const std::array<cv::Point2f, 6>& corners);
    static bool clusterEdges(std::vector<EdgeInfo>& edges);
    static std::vector<cv::Point2f> estimateVanishingPoints(const std::vector<EdgeInfo>& edges);
    static std::array<cv::Point2f, 6> rotateCorners(const std::array<cv::Point2f, 6>& corners, int offset, bool reverse);
    std::vector<cv::Point3f> buildCubeObjectPoints() const;
    PoseCandidate solvePoseFromHexagon(const std::array<cv::Point2f, 6>& corners) const;
    std::vector<cv::Point3f> transformToCameraCoordinates(const std::vector<cv::Point3f>& objectPoints,
                                                          const cv::Mat& rotationMatrix,
                                                          const cv::Vec3d& tvec) const;
    double computeReprojectionError(const std::vector<cv::Point3f>& objectPoints,
                                    const std::vector<cv::Point2f>& imagePoints,
                                    const cv::Vec3d& rvec,
                                    const cv::Vec3d& tvec) const;
    cv::Mat drawAnnotation(const cv::Mat& input,
                           const std::vector<cv::Point>& contour,
                           const std::array<cv::Point2f, 6>& hexagonCorners,
                           const std::vector<cv::Point2f>& projectedCorners,
                           const cv::Vec3d& rvec,
                           const cv::Vec3d& tvec,
                           double reprojectionError) const;
    static std::string vecToString(const cv::Vec3d& v);

    float originWidth_;
    float originHeight_;
    float originDepth_;
    double maxReprojectionErrorPx_ = 8.0;
    cv::Mat cameraMatrix_;
    cv::Mat distCoeffs_;
    int warpSize_;
};

}  // namespace tracker
