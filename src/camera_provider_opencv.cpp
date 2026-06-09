#include "camera_provider.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace camera {

namespace {

// 普通摄像头实现：通过 cv::VideoCapture 打开本地设备索引。
// 内参由构造时一次性加载/兜底，运行期间不变。
class OpenCVCameraProvider : public CameraProvider {
public:
    OpenCVCameraProvider(int cameraIndex, int width, int height, const Intrinsics& intrinsics)
        : intrinsics_(intrinsics), width_(width), height_(height) {
        cap_.open(cameraIndex);
        if (!cap_.isOpened()) {
            throw std::runtime_error("Could not open OpenCV camera index " +
                                     std::to_string(cameraIndex));
        }
        // 请求期望分辨率；实际分辨率由驱动决定，构造后回读
        cap_.set(cv::CAP_PROP_FRAME_WIDTH, width);
        cap_.set(cv::CAP_PROP_FRAME_HEIGHT, height);

        width_ = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
        height_ = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
        std::cout << "[OpenCVCameraProvider] opened camera " << cameraIndex
                  << " @ " << width_ << "x" << height_ << std::endl;
    }

    bool grab(FrameBundle& bundle) override {
        cap_ >> bundle.color;
        if (bundle.color.empty()) return false;
        bundle.depth = cv::Mat();  // 普通摄像头无深度
        bundle.intrinsics.cameraMatrix = intrinsics_.cameraMatrix.clone();
        bundle.intrinsics.distCoeffs = intrinsics_.distCoeffs.clone();
        return true;
    }

    int frameWidth() const override { return width_; }
    int frameHeight() const override { return height_; }
    std::string providerName() const override { return "opencv"; }

private:
    cv::VideoCapture cap_;
    Intrinsics intrinsics_;
    int width_;
    int height_;
};

// 文件模式实现：视频源可以是文件路径、RTSP/HTTP URL，或者退回到摄像头索引。
// 内参必须从 YAML 文件加载（不支持兜底，因为此模式专用于已知标定场景）。
class FileCameraProvider : public CameraProvider {
public:
    FileCameraProvider(const std::string& videoPath,
                       int fallbackCameraIndex,
                       int width,
                       int height,
                       const Intrinsics& intrinsics)
        : intrinsics_(intrinsics), width_(width), height_(height) {
        if (!videoPath.empty()) {
            // 优先用视频文件 / 流地址
            cap_.open(videoPath);
            if (!cap_.isOpened()) {
                throw std::runtime_error("Could not open video file: " + videoPath);
            }
            source_ = videoPath;
        } else {
            // 无 videoPath 时退回到摄像头索引
            cap_.open(fallbackCameraIndex);
            if (!cap_.isOpened()) {
                throw std::runtime_error("Could not open OpenCV camera index " +
                                         std::to_string(fallbackCameraIndex));
            }
            cap_.set(cv::CAP_PROP_FRAME_WIDTH, width);
            cap_.set(cv::CAP_PROP_FRAME_HEIGHT, height);
            source_ = "camera:" + std::to_string(fallbackCameraIndex);
        }

        width_ = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
        height_ = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
        std::cout << "[FileCameraProvider] source=" << source_
                  << " @ " << width_ << "x" << height_ << std::endl;
    }

    bool grab(FrameBundle& bundle) override {
        cap_ >> bundle.color;
        if (bundle.color.empty()) return false;
        bundle.depth = cv::Mat();
        bundle.intrinsics.cameraMatrix = intrinsics_.cameraMatrix.clone();
        bundle.intrinsics.distCoeffs = intrinsics_.distCoeffs.clone();
        return true;
    }

    int frameWidth() const override { return width_; }
    int frameHeight() const override { return height_; }
    std::string providerName() const override { return "file[" + source_ + "]"; }

private:
    cv::VideoCapture cap_;
    Intrinsics intrinsics_;
    int width_;
    int height_;
    std::string source_;
};

// 解析内参：有 YAML 路径就用真实标定，否则给警告 + identity 兜底
Intrinsics resolveIntrinsics(const CameraConfig& cfg) {
    if (!cfg.intrinsicsPath.empty()) {
        return loadIntrinsicsFromFile(cfg.intrinsicsPath);
    }
    std::cerr << "[CameraProvider] WARNING: no --intrinsics provided; "
                 "using identity intrinsics fallback (NOT calibrated for this device)."
              << std::endl;
    return makeDefaultIntrinsics(cfg.width, cfg.height);
}

}  // namespace

// 工厂入口（在 camera_provider.cpp 中通过前向声明调用）
std::unique_ptr<CameraProvider> makeOpenCVProvider(const CameraConfig& cfg) {
    Intrinsics intr = resolveIntrinsics(cfg);
    return std::make_unique<OpenCVCameraProvider>(cfg.cameraIndex, cfg.width, cfg.height, intr);
}

std::unique_ptr<CameraProvider> makeFileProvider(const CameraConfig& cfg) {
    Intrinsics intr = resolveIntrinsics(cfg);
    return std::make_unique<FileCameraProvider>(cfg.videoPath, cfg.cameraIndex,
                                                cfg.width, cfg.height, intr);
}

}  // namespace camera
