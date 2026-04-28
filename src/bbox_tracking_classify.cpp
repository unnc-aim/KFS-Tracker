#include "bbox_tracking_classify.h"

#include <torch/script.h>
#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>

namespace tracker {

namespace {

constexpr std::array<const char*, 32> kClassNames = {
    "R_R1",
    "B_R1",
    "T_03",
    "T_04",
    "T_05",
    "T_06",
    "T_07",
    "T_08",
    "T_09",
    "T_10",
    "T_11",
    "T_12",
    "T_13",
    "T_14",
    "T_15",
    "T_16",
    "T_17",
    "F_18",
    "F_19",
    "F_20",
    "F_21",
    "F_22",
    "F_23",
    "F_24",
    "F_25",
    "F_26",
    "F_27",
    "F_28",
    "F_29",
    "F_30",
    "F_31",
    "F_32",
};

inline float clamp01(float x) {
    return std::max(0.0F, std::min(1.0F, x));
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

cv::Rect decodeBboxToPixelRect(const torch::Tensor& bboxNorm, int width, int height) {
    const auto cpu = bboxNorm.to(torch::kCPU).contiguous();
    if (cpu.numel() != 4) {
        return cv::Rect();
    }

    const float x1n = clamp01(cpu[0].item<float>());
    const float y1n = clamp01(cpu[1].item<float>());
    const float x2n = clamp01(cpu[2].item<float>());
    const float y2n = clamp01(cpu[3].item<float>());

    int x1 = static_cast<int>(std::round(x1n * static_cast<float>(width)));
    int y1 = static_cast<int>(std::round(y1n * static_cast<float>(height)));
    int x2 = static_cast<int>(std::round(x2n * static_cast<float>(width)));
    int y2 = static_cast<int>(std::round(y2n * static_cast<float>(height)));

    x1 = std::max(0, std::min(x1, width - 1));
    y1 = std::max(0, std::min(y1, height - 1));
    x2 = std::max(0, std::min(x2, width));
    y2 = std::max(0, std::min(y2, height));

    if (x2 <= x1 || y2 <= y1) {
        return cv::Rect();
    }
    return cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
}

std::vector<cv::Point> rectToContour(const cv::Rect& rect) {
    if (rect.width <= 0 || rect.height <= 0) {
        return {};
    }
    return {
        rect.tl(),
        cv::Point(rect.x + rect.width, rect.y),
        rect.br(),
        cv::Point(rect.x, rect.y + rect.height),
    };
}

}  // namespace

class BboxTrackingClassifier::Impl {
public:
    explicit Impl(const std::string& modelPath) {
        module_ = torch::jit::load(modelPath, torch::kCPU);
        module_.eval();
    }

    torch::jit::Module module_;
};

BboxTrackingClassifier::BboxTrackingClassifier(const std::string& modelPath, int inputSize)
    : inputSize_(inputSize), impl_(std::make_shared<Impl>(modelPath)) {
    if (inputSize_ <= 0) {
        throw std::invalid_argument("inputSize must be positive.");
    }
}

BboxClassificationResult BboxTrackingClassifier::infer(const cv::Mat& image) const {
    BboxClassificationResult result;
    if (image.empty()) {
        return result;
    }

    torch::NoGradGuard noGrad;
    const torch::Tensor inputTensor = imageToInputTensor(image, inputSize_);

    const auto output = impl_->module_.forward({inputTensor});
    if (!output.isTuple()) {
        throw std::runtime_error("Model output is not a tuple of (bbox, class_logits).");
    }

    const auto elems = output.toTuple()->elements();
    if (elems.size() != 2 || !elems[0].isTensor() || !elems[1].isTensor()) {
        throw std::runtime_error("Unexpected model output structure.");
    }

    torch::Tensor bboxPred = elems[0].toTensor();
    torch::Tensor classLogits = elems[1].toTensor();
    if (bboxPred.dim() == 2 && bboxPred.size(0) == 1) {
        bboxPred = bboxPred.squeeze(0);
    }
    if (classLogits.dim() == 2 && classLogits.size(0) == 1) {
        classLogits = classLogits.squeeze(0);
    }

    const cv::Rect bbox = decodeBboxToPixelRect(bboxPred, image.cols, image.rows);
    auto probs = torch::softmax(classLogits, 0);
    const int classId = static_cast<int>(probs.argmax(0).item<int64_t>());
    const float classScore = probs[classId].item<float>();

    result.valid = bbox.width > 0 && bbox.height > 0;
    result.bbox = bbox;
    result.bboxContour = rectToContour(bbox);
    result.classId = classId;
    result.classScore = classScore;
    if (classId >= 0 && classId < static_cast<int>(kClassNames.size())) {
        result.className = kClassNames[static_cast<size_t>(classId)];
    } else {
        result.className = "unknown";
    }

    result.annotated = image.clone();
    if (result.valid) {
        cv::rectangle(result.annotated, result.bbox, cv::Scalar(0, 255, 0), 2);
        for (const auto& p : result.bboxContour) {
            cv::circle(result.annotated, p, 4, cv::Scalar(0, 255, 255), -1);
        }
    }
    const std::string label =
        "class: " + result.className + " (" + cv::format("id=%d, p=%.3f", result.classId, result.classScore) + ")";
    cv::putText(result.annotated, label, cv::Point(16, 32), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(50, 220, 50), 2);

    return result;
}

}  // namespace tracker
