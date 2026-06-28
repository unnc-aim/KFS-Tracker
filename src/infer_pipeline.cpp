#include "infer_pipeline.h"

namespace kfs {

// 推理 + 计时：对 variant 内的实际推理器调用一次 infer，把标注图、耗时与
// 定位结果（heatmap->corners，bbox->bbox）一并带出。
InferOutput inferAnnotated(Predictor &predictor, const cv::Mat &image) {
    InferOutput out;
    const double tickFreq = cv::getTickFrequency();

    std::visit(
        [&](auto &p) {
            using T = std::decay_t<decltype(p)>;
            const int64_t t0 = cv::getTickCount();
            auto r = p.infer(image);
            const int64_t t1 = cv::getTickCount();

            out.inferMs = static_cast<double>(t1 - t0) / tickFreq * 1000.0;
            out.valid = r.valid;
            out.annotated = r.annotated;

            if constexpr (std::is_same_v<T, tracker::HeatmapTrackingClassifier>) {
                out.isHeatmap = true;
                for (size_t i = 0; i < r.corners.size() && i < out.corners.size(); ++i) {
                    out.corners[i] = r.corners[i];
                }
                out.minConfidence = r.minConfidence;
                out.aspectRatio = r.aspectRatio;
            } else {
                out.isHeatmap = false;
                out.bbox = r.bbox;
            }
        },
        predictor);

    return out;
}

std::string formatClassification(const tracker::ClassifierInferResult &cls) {
    if (!cls.valid) {
        return "class: N/A";
    }
    return "class: " + cls.className + " (id=" + std::to_string(cls.classId) +
           ", p=" + cv::format("%.3f", cls.classScore) + ")";
}

void drawClassification(cv::Mat &annotated, const std::string &clsText) {
    if (annotated.empty()) {
        return;
    }
    cv::putText(annotated, clsText, cv::Point(20, 40),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
}

}  // namespace kfs
