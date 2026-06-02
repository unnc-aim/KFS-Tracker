#include "heatmap_classifier_infer.h"

#include <torch/script.h>
#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

namespace tracker {

namespace {

constexpr std::array<const char*, 32> kClassNames = {
    "R_R1", "B_R1", "T_03", "T_04", "T_05", "T_06", "T_07", "T_08",
    "T_09", "T_10", "T_11", "T_12", "T_13", "T_14", "T_15", "T_16",
    "T_17", "F_18", "F_19", "F_20", "F_21", "F_22", "F_23", "F_24",
    "F_25", "F_26", "F_27", "F_28", "F_29", "F_30", "F_31", "F_32",
};

inline float clamp01(float v) {
    return std::max(0.0f, std::min(1.0f, v));
}

torch::Tensor imageToInputTensor(const cv::Mat& image, int inputSize) {
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(inputSize, inputSize), 0, 0, cv::INTER_LINEAR);
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    cv::Mat rgbFloat;
    rgb.convertTo(rgbFloat, CV_32F, 1.0 / 255.0);

    auto tensor = torch::from_blob(
        rgbFloat.data, {inputSize, inputSize, 3}, torch::TensorOptions().dtype(torch::kFloat32)
    );
    tensor = tensor.permute({2, 0, 1}).unsqueeze(0).contiguous();
    return tensor.clone();
}

std::array<cv::Point2f, 4> orderPointsTLBLBRTR(const std::array<cv::Point2f, 4>& points) {
    std::vector<cv::Point2f> pts(points.begin(), points.end());
    std::sort(pts.begin(), pts.end(), [](const cv::Point2f& a, const cv::Point2f& b) { return a.x < b.x; });

    std::vector<cv::Point2f> left(pts.begin(), pts.begin() + 2);
    std::vector<cv::Point2f> right(pts.begin() + 2, pts.end());
    std::sort(left.begin(), left.end(), [](const cv::Point2f& a, const cv::Point2f& b) { return a.y < b.y; });
    std::sort(right.begin(), right.end(), [](const cv::Point2f& a, const cv::Point2f& b) { return a.y < b.y; });

    std::array<cv::Point2f, 4> out{};
    out[0] = left[0];   // tl
    out[1] = left[1];   // bl
    out[2] = right[1];  // br
    out[3] = right[0];  // tr
    return out;
}

}  // namespace

class HeatmapCornerInfer::Impl {
public:
    explicit Impl(const std::string& modelPath) {
        module_ = torch::jit::load(modelPath, torch::kCPU);
        module_.eval();
    }
    torch::jit::Module module_;
};

HeatmapCornerInfer::HeatmapCornerInfer(const std::string& modelPath, int inputSize)
    : inputSize_(inputSize), impl_(std::make_shared<Impl>(modelPath)) {
    if (inputSize_ <= 0) {
        throw std::invalid_argument("inputSize must be positive.");
    }
}

HeatmapInferResult HeatmapCornerInfer::infer(const cv::Mat& image) const {
    HeatmapInferResult result;
    if (image.empty()) {
        return result;
    }

    torch::NoGradGuard noGrad;
    torch::Tensor inputTensor = imageToInputTensor(image, inputSize_);
    auto output = impl_->module_.forward({inputTensor});
    if (!output.isTensor()) {
        throw std::runtime_error("Heatmap model output must be a tensor.");
    }

    torch::Tensor heatmaps = torch::sigmoid(output.toTensor());
    if (heatmaps.dim() != 4 || heatmaps.size(0) != 1 || heatmaps.size(1) < 4) {
        throw std::runtime_error("Unexpected heatmap output shape.");
    }

    heatmaps = heatmaps.squeeze(0).to(torch::kCPU);
    const int h = static_cast<int>(heatmaps.size(1));
    const int w = static_cast<int>(heatmaps.size(2));
    std::array<cv::Point2f, 4> corners{};

    for (int i = 0; i < 4; ++i) {
        torch::Tensor hm = heatmaps[i].contiguous().view({-1});
        int64_t idx = hm.argmax().item<int64_t>();
        int y = static_cast<int>(idx / w);
        int x = static_cast<int>(idx % w);

        float xNorm = static_cast<float>(x) / static_cast<float>(std::max(w - 1, 1));
        float yNorm = static_cast<float>(y) / static_cast<float>(std::max(h - 1, 1));
        corners[static_cast<size_t>(i)] = cv::Point2f(
            clamp01(xNorm) * static_cast<float>(image.cols - 1),
            clamp01(yNorm) * static_cast<float>(image.rows - 1)
        );
    }

    corners = orderPointsTLBLBRTR(corners);
    result.corners = corners;
    result.valid = true;

    result.annotated = image.clone();
    for (size_t i = 0; i < corners.size(); ++i) {
        cv::circle(result.annotated, corners[i], 5, cv::Scalar(0, 255, 255), -1);
        cv::putText(
            result.annotated,
            std::to_string(i),
            corners[i] + cv::Point2f(5.0f, -5.0f),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(0, 255, 255),
            1
        );
    }
    std::vector<cv::Point> poly;
    poly.reserve(4);
    for (const auto& p : corners) {
        poly.emplace_back(cvRound(p.x), cvRound(p.y));
    }
    cv::polylines(result.annotated, poly, true, cv::Scalar(0, 255, 0), 2);
    return result;
}

class ImageClassifierInfer::Impl {
public:
    explicit Impl(const std::string& modelPath) {
        module_ = torch::jit::load(modelPath, torch::kCPU);
        module_.eval();
    }
    torch::jit::Module module_;
};

ImageClassifierInfer::ImageClassifierInfer(const std::string& modelPath, int inputSize)
    : inputSize_(inputSize), impl_(std::make_shared<Impl>(modelPath)) {
    if (inputSize_ <= 0) {
        throw std::invalid_argument("inputSize must be positive.");
    }
}

ClassifierInferResult ImageClassifierInfer::infer(const cv::Mat& image) const {
    ClassifierInferResult result;
    if (image.empty()) {
        return result;
    }

    torch::NoGradGuard noGrad;
    torch::Tensor inputTensor = imageToInputTensor(image, inputSize_);
    auto output = impl_->module_.forward({inputTensor});
    if (!output.isTensor()) {
        throw std::runtime_error("Classifier model output must be a tensor.");
    }

    torch::Tensor logits = output.toTensor();
    if (logits.dim() == 2 && logits.size(0) == 1) {
        logits = logits.squeeze(0);
    }
    if (logits.dim() != 1) {
        throw std::runtime_error("Unexpected classifier output shape.");
    }

    torch::Tensor probs = torch::softmax(logits, 0);
    int classId = static_cast<int>(probs.argmax(0).item<int64_t>());
    float classScore = probs[classId].item<float>();

    result.valid = true;
    result.classId = classId;
    result.classScore = classScore;
    if (classId >= 0 && classId < static_cast<int>(kClassNames.size())) {
        result.className = kClassNames[static_cast<size_t>(classId)];
    } else {
        result.className = "unknown";
    }
    return result;
}

}  // namespace tracker

