#include "kfs_presence_detector.h"

#include "2D_tracker.h"

#include <cmath>
#include <vector>

namespace tracker {

// 近点有效距离区间，单位：米。后续调参直接改这里。
constexpr double kNearestPointMinDistanceM = 0.05;
constexpr double kNearestPointMaxDistanceM = 2.00;

namespace {

struct PlaneDepthStats {
    bool valid = false;
    double averageDistanceM = -1.0;
    int sampleCount = 0;
};

bool hasHighReprojectionLoss(const PlaneTracker2DResult& poseResult,
                             double reprojectionErrorThresholdPx) {
    return poseResult.reprojectionError < 0.0 ||
           !std::isfinite(poseResult.reprojectionError) ||
           poseResult.reprojectionError > reprojectionErrorThresholdPx;
}

std::array<cv::Point2f, 4> toCornerArray(const std::vector<cv::Point2f>& corners) {
    std::array<cv::Point2f, 4> result{};
    for (size_t i = 0; i < corners.size() && i < result.size(); ++i) {
        result[i] = corners[i];
    }
    return result;
}

PlaneDepthStats computePlaneAverageDistance(const cv::Mat& depth,
                                            const std::array<cv::Point2f, 4>& corners) {
    PlaneDepthStats stats;
    if (depth.empty() || depth.type() != CV_16U) {
        return stats;
    }

    std::vector<cv::Point> polygon;
    polygon.reserve(corners.size());
    for (const auto& corner : corners) {
        polygon.emplace_back(cvRound(corner.x), cvRound(corner.y));
    }

    cv::Rect roi = cv::boundingRect(polygon) & cv::Rect(0, 0, depth.cols, depth.rows);
    if (roi.empty()) {
        return stats;
    }

    cv::Mat mask = cv::Mat::zeros(depth.size(), CV_8U);
    cv::fillConvexPoly(mask, polygon, cv::Scalar(255), cv::LINE_AA);

    double distanceSumM = 0.0;
    int sampleCount = 0;
    for (int y = roi.y; y < roi.y + roi.height; ++y) {
        const uint16_t* depthRow = depth.ptr<uint16_t>(y);
        const uint8_t* maskRow = mask.ptr<uint8_t>(y);
        for (int x = roi.x; x < roi.x + roi.width; ++x) {
            if (maskRow[x] == 0 || depthRow[x] == 0) {
                continue;
            }
            const double distanceM = static_cast<double>(depthRow[x]) / 1000.0;
            distanceSumM += distanceM;
            ++sampleCount;
        }
    }

    if (sampleCount <= 0) {
        return stats;
    }

    stats.valid = true;
    stats.averageDistanceM = distanceSumM / static_cast<double>(sampleCount);
    stats.sampleCount = sampleCount;
    return stats;
}

void drawPresenceText(cv::Mat& annotated, const KFSPresenceCheckResult& result) {
    if (annotated.empty()) {
        return;
    }

    const cv::Scalar color = result.hasKfsCube ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
    cv::putText(annotated,
                result.hasKfsCube ? "KFS cube: yes" : "KFS cube: no",
                cv::Point(20, 30),
                cv::FONT_HERSHEY_SIMPLEX,
                0.75,
                color,
                2);

    if (result.reprojectionError >= 0.0) {
        cv::putText(annotated,
                    "reproj err: " + cv::format("%.3f px", result.reprojectionError),
                    cv::Point(20, 60),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.55,
                    cv::Scalar(255, 255, 0),
                    2);
    }

    if (result.depthValid) {
        cv::putText(annotated,
                    "plane depth: " + cv::format("%.3f m", result.averagePlaneDistanceM),
                    cv::Point(20, 88),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.55,
                    cv::Scalar(0, 255, 255),
                    2);
    }
}

}  // namespace

KFSPresenceCheckResult checkKfsCubeInFrame(const camera::FrameBundle& bundle,
                                           const HeatmapTrackingClassifier& cornerHeatmapTracker,
                                           const KFSPresenceCheckConfig& config) {
    KFSPresenceCheckResult result;

    if (bundle.color.empty()) {
        return result;
    }
    result.frameCaptured = true;
    result.annotated = bundle.color.clone();

    HeatmapTrackingResult heatmap = cornerHeatmapTracker.infer(bundle.color);
    result.heatmapValid = heatmap.valid;
    result.annotated = heatmap.annotated.empty() ? bundle.color.clone() : heatmap.annotated.clone();

    PlaneTracker2DResult pose;
    if (heatmap.valid && heatmap.corners.size() == result.corners.size()) {
        result.corners = toCornerArray(heatmap.corners);
        PlaneTracker2D planeTracker(config.planeWidth,
                                    config.planeHeight,
                                    bundle.intrinsics.cameraMatrix,
                                    bundle.intrinsics.distCoeffs,
                                    config.warpSize);
        pose = planeTracker.detectFromCorners(bundle.color, result.corners);
        result.poseSolved = pose.found;
        result.reprojectionError = pose.reprojectionError;
        if (!pose.annotated.empty()) {
            result.annotated = pose.annotated.clone();
        }
    }

    if (!heatmap.valid || heatmap.corners.size() != result.corners.size() ||
        hasHighReprojectionLoss(pose, config.reprojectionErrorThresholdPx)) {
        result.hasKfsCube = false;
        drawPresenceText(result.annotated, result);
        return result;
    }

    PlaneDepthStats depthStats = computePlaneAverageDistance(bundle.depth, result.corners);
    result.depthValid = depthStats.valid;
    result.averagePlaneDistanceM = depthStats.averageDistanceM;
    result.validDepthSamples = depthStats.sampleCount;
    result.hasKfsCube = depthStats.valid &&
                        depthStats.averageDistanceM >= kNearestPointMinDistanceM &&
                        depthStats.averageDistanceM <= kNearestPointMaxDistanceM;
    drawPresenceText(result.annotated, result);
    return result;
}

KFSPresenceCheckResult checkKfsCubeInCamera(camera::CameraProvider& provider,
                                            const HeatmapTrackingClassifier& cornerHeatmapTracker,
                                            const KFSPresenceCheckConfig& config) {
    camera::FrameBundle bundle;
    if (!provider.grab(bundle)) {
        return {};
    }
    return checkKfsCubeInFrame(bundle, cornerHeatmapTracker, config);
}

bool hasKfsCubeInCamera(camera::CameraProvider& provider,
                        const HeatmapTrackingClassifier& cornerHeatmapTracker,
                        const KFSPresenceCheckConfig& config) {
    return checkKfsCubeInCamera(provider, cornerHeatmapTracker, config).hasKfsCube;
}

}  // namespace tracker
