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

// ---- 热力图解码：对每个通道 argmax 得到归一化角点坐标 ----
// 输入: heatmaps (1, 4, H, W)
// 输出: 4 个归一化角点 (x,y) ∈ [0,1]，顺序为 tl, bl, br, tr
std::vector<cv::Point2f> decodeCornersFromHeatmaps(
    const torch::Tensor& heatmaps, int origW, int origH) {
    // heatmaps: (1, 4, H, W)
    const int c = heatmaps.size(1);
    const int h = heatmaps.size(2);
    const int w = heatmaps.size(3);

    std::vector<cv::Point2f> corners;
    corners.reserve(static_cast<size_t>(c));

    for (int ch = 0; ch < c; ++ch) {
        auto channel = heatmaps[0][ch];  // (H, W)
        auto flat = channel.reshape(-1);
        int64_t idx = flat.argmax(0).item<int64_t>();
        int64_t y = idx / w;
        int64_t x = idx % w;

        float xNorm = static_cast<float>(x) / std::max(w - 1, 1);
        float yNorm = static_cast<float>(y) / std::max(h - 1, 1);

        corners.emplace_back(xNorm * static_cast<float>(origW),
                             yNorm * static_cast<float>(origH));
    }
    return corners;
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

    // 解码角点
    auto corners = decodeCornersFromHeatmaps(heatmaps, image.cols, image.rows);
    orderCorners(corners);

    result.valid = corners.size() == 4;
    result.corners = std::move(corners);

    // 可视化（不计时）
    result.annotated = image.clone();
    annotateResult(result.annotated, result.corners);

    return result;
}

}  // namespace tracker
