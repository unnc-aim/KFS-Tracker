#pragma once

#include <opencv2/opencv.hpp>

#include <array>
#include <string>
#include <variant>

#include "bbox_tracking_classify.h"
#include "heatmap_classifier_infer.h"
#include "heatmap_tracking_classify.h"

// 共用推理管线：把 main.cpp 原本的推理 + 计时 + 绘制逻辑提升为库符号，
// 供 CLI（kfs_tracker）与 ROS 节点（kfs_tracker_node）共用，避免逻辑重复。
namespace kfs {

// 推理器 variant（bbox / heatmap 二选一）。
using Predictor = std::variant<tracker::BboxTrackingClassifier,
                               tracker::HeatmapTrackingClassifier>;

// 单帧推理 + 计时输出：携带标注图、耗时，以及定位结果（角点或 bbox）。
// 关键：一次推理即得到 corners/bbox，消费方无需重复调用 predictor.infer()。
struct InferOutput {
    bool valid = false;
    cv::Mat annotated;
    double inferMs = 0.0;
    bool isHeatmap = false;                 // 区分定位结果取 corners 还是 bbox
    cv::Rect bbox;                          // bbox 模式有效
    std::array<cv::Point2f, 4> corners{};   // heatmap 模式有效（TL, BL, BR, TR）
    float minConfidence = 0.0f;             // heatmap: 4 角点热力图峰值最小值
    float aspectRatio = 0.0f;               // heatmap: 四边形最大边/最小边
};

// 单次推理 + 计时，返回标注图与定位结果（不向 stdout 打印）。
InferOutput inferAnnotated(Predictor &predictor, const cv::Mat &image);

// 分类结果文本格式化："class: NAME (id=X, p=0.XXX)" / "class: N/A"
std::string formatClassification(const tracker::ClassifierInferResult &cls);

// 在标注图左上角叠加分类文本。
void drawClassification(cv::Mat &annotated, const std::string &clsText);

}  // namespace kfs
