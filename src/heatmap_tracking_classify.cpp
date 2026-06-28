#include "heatmap_tracking_classify.h"

#include <torch/script.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <stdexcept>

namespace tracker {

namespace {

// ---- 预处理：BGR Mat → (1,3,H,W) float [0,1] ----
torch::Tensor imageToInputTensor(const cv::Mat& image, int inputSize) {
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(inputSize, inputSize), 0, 0, cv::INTER_LINEAR);
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    cv::Mat rgbFloat;
    rgb.convertTo(rgbFloat, CV_32F, 1.0 / 255.0);
    auto tensor = torch::from_blob(
        rgbFloat.data, {inputSize, inputSize, 3}, torch::TensorOptions().dtype(torch::kFloat32));
    tensor = tensor.permute({2, 0, 1}).unsqueeze(0).contiguous();
    return tensor.clone();
}

// ---- 热力图解码：对每个通道 argmax 得到归一化角点坐标 + 峰值置信度 ----
// 输入: heatmaps (1, 4, H, W)
// 输出: 4 个归一化角点 (x,y) ∈ [0,1]，顺序为 tl, bl, br, tr
//       confidences: 每个角点通道的峰值（sigmoid 后的 0~1 值）
std::vector<cv::Point2f> decodeCornersFromHeatmaps(
    const torch::Tensor& heatmaps, int origW, int origH,
    std::vector<float>& confidences) {
    // heatmaps: (1, 4, H, W)
    const int c = heatmaps.size(1);
    const int h = heatmaps.size(2);
    const int w = heatmaps.size(3);

    std::vector<cv::Point2f> corners;
    corners.reserve(static_cast<size_t>(c));
    confidences.clear();
    confidences.reserve(static_cast<size_t>(c));

    for (int ch = 0; ch < c; ++ch) {
        auto channel = heatmaps[0][ch];  // (H, W)
        auto flat = channel.reshape(-1);
        auto maxTuple = flat.max(0);
        int64_t idx = std::get<1>(maxTuple).item<int64_t>();
        float peakVal = std::get<0>(maxTuple).item<float>();

        // 如果热力图未过 sigmoid，用 sigmoid 归一化到 [0,1]
        float conf = 1.0f / (1.0f + std::exp(-peakVal));
        confidences.push_back(conf);

        int64_t y = idx / w;
        int64_t x = idx % w;

        float xNorm = static_cast<float>(x) / std::max(w - 1, 1);
        float yNorm = static_cast<float>(y) / std::max(h - 1, 1);

        corners.emplace_back(xNorm * static_cast<float>(origW),
                             yNorm * static_cast<float>(origH));
    }
    return corners;
}

// ---- 几何合理性检查：计算四边形最大边长 / 最小边长 ----
// 返回值 > 1.0；值越大说明角点越不合理（某个角点飞了）
float computeAspectRatio(const std::vector<cv::Point2f>& corners) {
    if (corners.size() != 4) return 9999.0f;
    // corners 顺序: tl, bl, br, tr（排序后）
    // 四条边: tl-bl, bl-br, br-tr, tr-tl
    float edges[4];
    edges[0] = static_cast<float>(cv::norm(corners[0] - corners[1])); // tl-bl
    edges[1] = static_cast<float>(cv::norm(corners[1] - corners[2])); // bl-br
    edges[2] = static_cast<float>(cv::norm(corners[2] - corners[3])); // br-tr
    edges[3] = static_cast<float>(cv::norm(corners[3] - corners[0])); // tr-tl

    float minE = *std::min_element(edges, edges + 4);
    float maxE = *std::max_element(edges, edges + 4);
    if (minE < 1.0f) return 9999.0f;  // 面积太小，不合理
    return maxE / minE;
}

// ---- 角点排序：左上、左下、右下、右上 ----
void orderCorners(std::vector<cv::Point2f>& corners) {
    if (corners.size() != 4) return;

    // 按 x 排序，分左右
    std::sort(corners.begin(), corners.end(),
              [](const cv::Point2f& a, const cv::Point2f& b) { return a.x < b.x; });

    // 左侧两个按 y 排序（y 小的在上）
    std::sort(corners.begin(), corners.begin() + 2,
              [](const cv::Point2f& a, const cv::Point2f& b) { return a.y < b.y; });
    // 右侧两个按 y 排序
    std::sort(corners.begin() + 2, corners.end(),
              [](const cv::Point2f& a, const cv::Point2f& b) { return a.y < b.y; });

    // 现在 corners = [tl, bl, tr, br]，重排为 tl, bl, br, tr
    std::swap(corners[2], corners[3]);
}

// ---- 可视化 ----
void annotateResult(
    cv::Mat& img,
    const std::vector<cv::Point2f>& corners) {
    if (corners.size() != 4) return;

    // 画四边形连线
    std::vector<cv::Point> pts;
    for (const auto& p : corners) {
        pts.emplace_back(static_cast<int>(std::round(p.x)),
                         static_cast<int>(std::round(p.y)));
    }
    // tl → bl → br → tr → tl
    for (int i = 0; i < 4; ++i) {
        cv::line(img, pts[i], pts[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
    }

    // 画角点（4 种颜色区分）
    const cv::Scalar colors[] = {
        {0, 0, 255},    // tl - 红
        {255, 0, 0},    // bl - 蓝
        {255, 255, 0},  // br - 青
        {0, 255, 255},  // tr - 黄
    };
    const char* names[] = {"TL", "BL", "BR", "TR"};
    for (int i = 0; i < 4; ++i) {
        cv::circle(img, pts[i], 5, colors[i], -1);
        cv::putText(img, names[i], pts[i] + cv::Point(6, -6),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, colors[i], 1);
    }
    // 推理耗时与分类文本统一由 main 绘制，此处仅保留角点几何标注。
}

}  // namespace

// ---- Impl ----
class HeatmapTrackingClassifier::Impl {
public:
    explicit Impl(const std::string& modelPath) {
        module_ = torch::jit::load(modelPath, torch::kCPU);
        module_.eval();
    }

    torch::jit::Module module_;
};

HeatmapTrackingClassifier::HeatmapTrackingClassifier(const std::string& modelPath, int inputSize)
    : inputSize_(inputSize), impl_(std::make_shared<Impl>(modelPath)) {
    if (inputSize_ <= 0) {
        throw std::invalid_argument("inputSize must be positive.");
    }
}

HeatmapTrackingResult HeatmapTrackingClassifier::infer(const cv::Mat& image) const {
    HeatmapTrackingResult result;
    if (image.empty()) {
        return result;
    }

    torch::NoGradGuard noGrad;
    const torch::Tensor inputTensor = imageToInputTensor(image, inputSize_);

    // 前向推理：输出 (1, 4, H, W) 热力图
    const auto output = impl_->module_.forward({inputTensor});
    if (!output.isTensor()) {
        throw std::runtime_error("Heatmap model output is not a tensor.");
    }
    torch::Tensor heatmaps = output.toTensor();

    // 解码角点 + 置信度
    std::vector<float> confidences;
    auto corners = decodeCornersFromHeatmaps(heatmaps, image.cols, image.rows, confidences);
    orderCorners(corners);

    // ---- 筛选第一层：热力图峰值置信度 ----
    // 4 个角点中最小的置信度低于阈值 → 判定无效
    result.minConfidence = *std::min_element(confidences.begin(), confidences.end());
    constexpr float kMinConfidence = 0.3f;  // sigmoid 后的阈值

    // ---- 筛选第二层：几何合理性 ----
    // 四边形最大边长 / 最小边长过大 → 某个角点飞了
    result.aspectRatio = computeAspectRatio(corners);
    constexpr float kMaxAspectRatio = 5.0f;

    result.corners = std::move(corners);
    result.valid = (result.minConfidence >= kMinConfidence &&
                    result.aspectRatio <= kMaxAspectRatio);

    // 可视化（不计时）
    result.annotated = image.clone();
    annotateResult(result.annotated, result.corners);

    return result;
}

}  // namespace tracker
