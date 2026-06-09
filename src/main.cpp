#include "bbox_tracking_classify.h"
#include "camera_provider.h"
#include "heatmap_classifier_infer.h"
#include "heatmap_tracking_classify.h"
#include <opencv2/opencv.hpp>

#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>
#include <variant>
#include <vector>

// 命令行参数解析：查找 --key 后面的值，找不到则返回 def
static std::string getArgValue(int argc, char** argv, const std::string& key, const std::string& def) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (key == argv[i]) {
            return argv[i + 1];
        }
    }
    return def;
}

// 检查环境变量 DISABLE_RENDERER，默认弹窗渲染
static bool shouldRenderOutput()
{
    const char *env = std::getenv("DISABLE_RENDERER");
    if (env == nullptr)
        return true;
    std::string val(env);
    return val.empty() || val == "0" || val == "false" || val == "FALSE";
}

// 推理器 variant 类型
using Predictor = std::variant<tracker::BboxTrackingClassifier, tracker::HeatmapTrackingClassifier>;

// 单帧推理结果输出到 stdout（真实推理耗时 + 理论帧率）
static void printInferenceResult(const tracker::BboxClassificationResult &result, int frameIndex, double inferMs)
{
    double inferFps = inferMs > 0.0 ? 1000.0 / inferMs : 0.0;
    if (!result.valid)
    {
        std::cout << "[frame " << frameIndex << "] infer: " << cv::format("%.1f", inferMs) << "ms"
                  << " (" << cv::format("%.1f", inferFps) << " fps)"
                  << " | bbox invalid/empty." << std::endl;
    }
    else
    {
        // 分类已由 ImageClassifierInfer(tracker_classifier.pt) 统一接管，此处仅输出 bbox 定位
        std::cout << "[frame " << frameIndex << "] infer: " << cv::format("%.1f", inferMs) << "ms"
                  << " (" << cv::format("%.1f", inferFps) << " fps)"
                  << " | bbox: " << result.bbox
                  << std::endl;
    }
}

static void printInferenceResult(const tracker::HeatmapTrackingResult &result, int frameIndex, double inferMs)
{
    double inferFps = inferMs > 0.0 ? 1000.0 / inferMs : 0.0;
    if (!result.valid)
    {
        std::cout << "[frame " << frameIndex << "] infer: " << cv::format("%.1f", inferMs) << "ms"
                  << " (" << cv::format("%.1f", inferFps) << " fps)"
                  << " | corners not detected." << std::endl;
    }
    else
    {
        std::cout << "[frame " << frameIndex << "] infer: " << cv::format("%.1f", inferMs) << "ms"
                  << " (" << cv::format("%.1f", inferFps) << " fps)"
                  << " | corners:";
        for (size_t i = 0; i < result.corners.size(); ++i)
        {
            std::cout << " (" << result.corners[i].x << "," << result.corners[i].y << ")";
        }
        std::cout << std::endl;
    }
}

// 统一推理 + 计时 + 打印 + 返回标注图和耗时
struct InferOutputWithTime {
    bool valid = false;
    cv::Mat annotated;
    double inferMs = 0.0;
};

static InferOutputWithTime inferPrintAnnotated(Predictor &predictor, const cv::Mat &image,
                                                int frameIndex) {
    InferOutputWithTime out;
    const double tickFreq = cv::getTickFrequency();

    std::visit([&](auto &p) {
        using T = std::decay_t<decltype(p)>;
        int64_t t0 = cv::getTickCount();
        if constexpr (std::is_same_v<T, tracker::BboxTrackingClassifier>) {
            auto r = p.infer(image);
            int64_t t1 = cv::getTickCount();
            out.inferMs = static_cast<double>(t1 - t0) / tickFreq * 1000.0;
            printInferenceResult(r, frameIndex, out.inferMs);
            out.valid = r.valid;
            out.annotated = r.annotated;
        } else {
            auto r = p.infer(image);
            int64_t t1 = cv::getTickCount();
            out.inferMs = static_cast<double>(t1 - t0) / tickFreq * 1000.0;
            printInferenceResult(r, frameIndex, out.inferMs);
            out.valid = r.valid;
            out.annotated = r.annotated;
        }
    }, predictor);
    return out;
}

// 分类结果统一由 ImageClassifierInfer(tracker_classifier.pt) 接管
static std::string formatClassification(const tracker::ClassifierInferResult &cls)
{
    if (!cls.valid)
        return "class: N/A";
    return "class: " + cls.className + " (id=" + std::to_string(cls.classId)
           + ", p=" + cv::format("%.3f", cls.classScore) + ")";
}

// 在标注图左上角叠加分类文本
static void drawClassification(cv::Mat &annotated, const std::string &clsText)
{
    if (annotated.empty())
        return;
    cv::putText(annotated, clsText, cv::Point(20, 40),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
}

// 摄像头/视频流模式：通过 CameraProvider 抽象层采集帧
static int runCamera(Predictor &predictor,
                     tracker::ImageClassifierInfer &classifierInfer,
                     std::unique_ptr<camera::CameraProvider> provider,
                     bool render)
{
    const std::string windowName = "KFS-Tracker";
    if (render)
    {
        cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
    }

    camera::FrameBundle bundle;
    int frameIndex = 0;
    double fps = 0.0;

    std::cout << "Camera stream started (" << provider->providerName()
              << ", " << provider->frameWidth() << "x" << provider->frameHeight() << "). "
              << "Press 'q' or ESC to quit." << std::endl;

    while (true)
    {
        if (!provider->grab(bundle))
        {
            std::cerr << "[frame " << frameIndex << "] empty frame, stopping." << std::endl;
            break;
        }

        // 推理 + 打印（内部计时）
        auto inferOut = inferPrintAnnotated(predictor, bundle.color, frameIndex);
        double inferMs = inferOut.inferMs;
        double inferFps = inferMs > 0.0 ? 1000.0 / inferMs : 0.0;
        fps = fps * 0.9 + inferFps * 0.1;

        // 分类统一由 ImageClassifierInfer(tracker_classifier.pt) 接管
        tracker::ClassifierInferResult clsResult = classifierInfer.infer(bundle.color);
        std::string clsText = formatClassification(clsResult);
        drawClassification(inferOut.annotated, clsText);
        std::cout << "  " << clsText << std::endl;

        // bundle.intrinsics 后续可传给 CubeTracker3D / PlaneTracker2D 做位姿估计

        if (render)
        {
            int y = inferOut.annotated.rows - 20;
            cv::putText(inferOut.annotated,
                        cv::format("Infer: %.1f ms", inferMs),
                        cv::Point(20, y - 28),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7,
                        cv::Scalar(0, 255, 255), 2);
            cv::putText(inferOut.annotated,
                        cv::format("Max FPS: %.1f (EMA %.1f)", inferFps, fps),
                        cv::Point(20, y),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7,
                        cv::Scalar(0, 255, 0), 2);

            cv::imshow(windowName, inferOut.annotated);
            int key = cv::waitKey(1);
            if (key == 'q' || key == 27)
            {
                std::cout << "Quit signal received." << std::endl;
                break;
            }
        }

        ++frameIndex;
    }

    if (render)
    {
        cv::destroyWindow(windowName);
    }
    std::cout << "Camera stream ended after " << frameIndex << " frames." << std::endl;
    return 0;
}

// 单图模式
static int runImage(Predictor &predictor,
                    tracker::ImageClassifierInfer &classifierInfer,
                    const std::string &imagePath,
                    const std::string &outputPath,
                    bool render)
{
    cv::Mat img = cv::imread(imagePath);
    if (img.empty())
    {
        std::cerr << "Could not open image: " << imagePath << std::endl;
        return -1;
    }

    // 推理（定位/角点）；分类统一由 ImageClassifierInfer(tracker_classifier.pt) 接管
    cv::Mat annotated;
    std::visit([&](auto &p) {
        using T = std::decay_t<decltype(p)>;
        if constexpr (std::is_same_v<T, tracker::BboxTrackingClassifier>) {
            tracker::BboxClassificationResult result = p.infer(img);
            if (!result.valid) {
                std::cout << "Inference finished, but bbox is invalid/empty." << std::endl;
            } else {
                std::cout << "Bbox contour points:" << std::endl;
                for (size_t i = 0; i < result.bboxContour.size(); ++i) {
                    std::cout << "  [" << i << "] " << result.bboxContour[i] << std::endl;
                }
            }
            annotated = result.annotated;
        } else {
            tracker::HeatmapTrackingResult result = p.infer(img);
            if (!result.valid) {
                std::cout << "Inference finished, but corners not detected." << std::endl;
            } else {
                std::cout << "Corner points:" << std::endl;
                const char* names[] = {"TL", "BL", "BR", "TR"};
                for (size_t i = 0; i < result.corners.size(); ++i) {
                    std::cout << "  [" << names[i] << "] (" << result.corners[i].x
                              << ", " << result.corners[i].y << ")" << std::endl;
                }
            }
            annotated = result.annotated;
        }
    }, predictor);

    // 分类（统一接管：bbox 与 heatmap 模式都走 tracker_classifier.pt）
    tracker::ClassifierInferResult clsResult = classifierInfer.infer(img);
    std::string clsText = formatClassification(clsResult);
    drawClassification(annotated, clsText);
    std::cout << "Classification: " << clsText << std::endl;

    cv::imwrite(outputPath, annotated);
    std::cout << "Saved: " << outputPath << std::endl;

    if (render)
    {
        cv::Mat annotatedImg = cv::imread(outputPath);
        const std::string windowName = "KFS-Tracker";
        cv::namedWindow(windowName, cv::WINDOW_NORMAL);
        cv::imshow(windowName, annotatedImg);
        std::cout << "Press any key to close window..." << std::endl;
        cv::waitKey(0);
        cv::destroyWindow(windowName);
    }

    return 0;
}

// 从命令行参数构造 CameraConfig
static camera::CameraConfig buildCameraConfig(int argc, char **argv)
{
    camera::CameraConfig cfg;
    cfg.type = getArgValue(argc, argv, "--camera_type", "opencv");
    cfg.intrinsicsPath = getArgValue(argc, argv, "--intrinsics", "");
    cfg.cameraIndex = std::stoi(getArgValue(argc, argv, "--camera_index", "0"));
    cfg.videoPath = getArgValue(argc, argv, "--video", "");
    cfg.width = std::stoi(getArgValue(argc, argv, "--width", "1280"));
    cfg.height = std::stoi(getArgValue(argc, argv, "--height", "720"));
    return cfg;
}

int main(int argc, char **argv)
{
    try
    {
        // --help 作为独立标志，没有后续值，不能通过 getArgValue 检测
        bool showHelp = (argc < 2);
        for (int i = 1; i < argc; ++i)
        {
            if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h")
            {
                showHelp = true;
                break;
            }
        }
        if (showHelp)
        {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "\n"
                      << "Modes:\n"
                      << "  --image <path>          Single-image mode (default: image.jpg)\n"
                      << "  --camera_type <type>    Camera stream mode; types: opencv | realsense | file\n"
                      << "\n"
                      << "Camera options:\n"
                      << "  --camera_index <int>    Camera device index (default: 0)\n"
                      << "  --intrinsics <path>     Intrinsics YAML file (OpenCV calibration format)\n"
                      << "  --video <path>          Video source for --camera_type file\n"
                      << "  --width <int>           Desired frame width (default: 1280)\n"
                      << "  --height <int>          Desired frame height (default: 720)\n"
                      << "\n"
                      << "Inference options:\n"
                      << "  --model_type <type>     Model type: heatmap | bbox (default: heatmap)\n"
                      << "  --output <path>         Output annotated image path (default: annotated.jpg)\n"
                      << "  --model <path>          Model file path (default follows --model_type)\n"
                      << "  --input_size <int>      Model input size (default: 300)\n"
                      << "  --classifier_model <path> Classifier model (default: model/tracker_classifier.pt)\n"
                      << "\n"
                      << "Environment:\n"
                      << "  DISABLE_RENDERER=1      Suppress OpenCV GUI window\n";
            return 0;
        }

        bool render = shouldRenderOutput();
        std::string modelType = getArgValue(argc, argv, "--model_type", "heatmap");
        // 默认模型路径跟随 modelType：heatmap 用角点热力图模型，bbox 用回归模型
        std::string defaultModelPath = (modelType == "heatmap")
            ? "model/corner_heatmap_localizer.pt"
            : "model/tracker_localizer.pt";
        std::string modelPath = getArgValue(argc, argv, "--model", defaultModelPath);
        std::string classifierModelPath = getArgValue(argc, argv, "--classifier_model", "model/tracker_classifier.pt");
        int inputSize = std::stoi(getArgValue(argc, argv, "--input_size", "300"));

        // 分类统一由 tracker_classifier.pt 接管（bbox / heatmap 模式共用）
        tracker::ImageClassifierInfer classifierInfer(classifierModelPath, inputSize);

        Predictor predictor = [&]() -> Predictor {
            if (modelType == "heatmap") {
                return tracker::HeatmapTrackingClassifier(modelPath, inputSize);
            }
            return tracker::BboxTrackingClassifier(modelPath, inputSize);
        }();

        // 判断是摄像头流模式还是单图模式
        std::string cameraType = getArgValue(argc, argv, "--camera_type", "");
        if (!cameraType.empty())
        {
            camera::CameraConfig cfg = buildCameraConfig(argc, argv);
            cfg.type = cameraType;
            auto provider = camera::createCameraProvider(cfg);
            if (!provider)
            {
                return -1;
            }
            return runCamera(predictor, classifierInfer, std::move(provider), render);
        }

        // 单图模式
        std::string imagePath = getArgValue(argc, argv, "--image", "image.jpg");
        std::string outputPath = getArgValue(argc, argv, "--output", "annotated.jpg");
        return runImage(predictor, classifierInfer, imagePath, outputPath, render);
    }
    catch (const cv::Exception &e)
    {
        std::cerr << "[OpenCV][main] " << e.what() << std::endl;
        return -1;
    }
    catch (const std::exception &e)
    {
        std::cerr << "[std::exception][main] " << e.what() << std::endl;
        return -1;
    }
}
