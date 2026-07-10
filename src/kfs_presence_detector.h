#pragma once

#include "camera_provider.h"
#include "heatmap_tracking_classify.h"

#include <opencv2/opencv.hpp>

#include <array>
#include <string>

namespace tracker {

struct KFSPresenceCheckConfig {
    double reprojectionErrorThresholdPx = 8.0;
    float planeWidth = 1.0f;
    float planeHeight = 1.0f;
    int warpSize = 224;
};

struct KFSPresenceCheckResult {
    bool hasKfsCube = false;
    bool frameCaptured = false;
    bool heatmapValid = false;
    bool poseSolved = false;
    bool depthValid = false;
    double reprojectionError = -1.0;
    double averagePlaneDistanceM = -1.0;
    int validDepthSamples = 0;
    std::array<cv::Point2f, 4> corners{};
    cv::Mat annotated;
};

// 从相机抓取一帧，用 corner_heatmap_tracker 追踪角点并计算重投影误差。
// 重投影误差过高直接返回 false；误差合格时，再用深度图计算追踪平面平均距离，
// 平均距离超过近点距离上限返回 false，否则返回 true。
KFSPresenceCheckResult checkKfsCubeInFrame(const camera::FrameBundle& bundle,
                                           const HeatmapTrackingClassifier& cornerHeatmapTracker,
                                           const KFSPresenceCheckConfig& config = KFSPresenceCheckConfig{});

KFSPresenceCheckResult checkKfsCubeInCamera(camera::CameraProvider& provider,
                                            const HeatmapTrackingClassifier& cornerHeatmapTracker,
                                            const KFSPresenceCheckConfig& config = KFSPresenceCheckConfig{});

bool hasKfsCubeInCamera(camera::CameraProvider& provider,
                        const HeatmapTrackingClassifier& cornerHeatmapTracker,
                        const KFSPresenceCheckConfig& config = KFSPresenceCheckConfig{});

}  // namespace tracker
