// 整个文件条件编译：仅在 ENABLE_REALSENSE 定义时才参与构建。
// 这样没有装 librealsense2 的机器也能正常构建其他模式。

#ifdef ENABLE_REALSENSE

#include "camera_provider.h"

#include <librealsense2/rs.hpp>

#include <iostream>
#include <stdexcept>
#include <string>

namespace camera {

namespace {

// 把 RealSense SDK 的 rs2_intrinsics 转换为 OpenCV 风格的内参。
//   fx, fy, ppx, ppy -> 3x3 相机矩阵
//   coeffs[0..4]     -> 1x5 Brown-Conrady 畸变系数 (k1, k2, p1, p2, k3)
Intrinsics convertIntrinsics(const rs2_intrinsics& intr) {
    Intrinsics result;
    result.cameraMatrix = (cv::Mat_<double>(3, 3) <<
        static_cast<double>(intr.fx), 0.0,                              static_cast<double>(intr.ppx),
        0.0,                              static_cast<double>(intr.fy), static_cast<double>(intr.ppy),
        0.0,                              0.0,                              1.0);

    cv::Mat dist(1, 5, CV_64F);
    for (int i = 0; i < 5; ++i) {
        dist.at<double>(0, i) = static_cast<double>(intr.coeffs[i]);
    }
    result.distCoeffs = dist;
    return result;
}

// RealSense 实现：通过 rs2::pipeline 采集对齐到彩色流的 color + depth 帧。
class RealSenseCameraProvider : public CameraProvider {
public:
    RealSenseCameraProvider(int width, int height, int fps) {
        // 启动 color + depth 两路流；后续在 grab() 中对齐到 color 帧
        cfg_.enable_stream(RS2_STREAM_COLOR, width, height, RS2_FORMAT_BGR8, fps);
        cfg_.enable_stream(RS2_STREAM_DEPTH, width, height, RS2_FORMAT_Z16, fps);

        profile_ = pipe_.start(cfg_);

        // 从 color profile 拿到内参；RealSense 设备自身已校准，无需 YAML
        auto color_profile = profile_.get_stream(RS2_STREAM_COLOR)
                                 .as<rs2::video_stream_profile>();
        rs2_intrinsics color_intr = color_profile.get_intrinsics();
        intrinsics_ = convertIntrinsics(color_intr);

        width_ = color_profile.width();
        height_ = color_profile.height();

        std::cout << "[RealSenseCameraProvider] started @ "
                  << width_ << "x" << height_ << std::endl;
        std::cout << "  fx=" << intrinsics_.cameraMatrix.at<double>(0, 0)
                  << " fy=" << intrinsics_.cameraMatrix.at<double>(1, 1)
                  << " cx=" << intrinsics_.cameraMatrix.at<double>(0, 2)
                  << " cy=" << intrinsics_.cameraMatrix.at<double>(1, 2)
                  << std::endl;
    }

    bool grab(FrameBundle& bundle) override {
        rs2::frameset frames;
        try {
            // 5s 超时，避免设备掉线时无限阻塞
            frames = pipe_.wait_for_frames(5000);
        } catch (const rs2::error& e) {
            std::cerr << "[RealSense] wait_for_frames failed: " << e.what() << std::endl;
            return false;
        }

        // 把深度帧对齐到彩色帧的像素坐标系，两者分辨率一致
        rs2::align align_to_color(RS2_STREAM_COLOR);
        frames = align_to_color.process(frames);

        rs2::video_frame color_frame = frames.get_color_frame();
        rs2::depth_frame depth_frame = frames.get_depth_frame();

        // 用 rs2 帧的内存构造 cv::Mat，再 clone() 一次拷贝出来，避免数据被复用覆盖
        bundle.color = cv::Mat(
            cv::Size(color_frame.get_width(), color_frame.get_height()),
            CV_8UC3,
            const_cast<void*>(color_frame.get_data()),
            cv::Mat::AUTO_STEP).clone();

        bundle.depth = cv::Mat(
            cv::Size(depth_frame.get_width(), depth_frame.get_height()),
            CV_16UC1,
            const_cast<void*>(depth_frame.get_data()),
            cv::Mat::AUTO_STEP).clone();

        bundle.intrinsics.cameraMatrix = intrinsics_.cameraMatrix.clone();
        bundle.intrinsics.distCoeffs = intrinsics_.distCoeffs.clone();
        return true;
    }

    int frameWidth() const override { return width_; }
    int frameHeight() const override { return height_; }
    std::string providerName() const override { return "realsense"; }

private:
    rs2::pipeline pipe_;
    rs2::config cfg_;
    rs2::pipeline_profile profile_;
    Intrinsics intrinsics_;
    int width_ = 0;
    int height_ = 0;
};

}  // namespace

// 工厂入口（camera_provider.cpp 通过前向声明调用）
std::unique_ptr<CameraProvider> makeRealSenseProvider(const CameraConfig& cfg) {
    const int fps = 30;
    return std::make_unique<RealSenseCameraProvider>(cfg.width, cfg.height, fps);
}

}  // namespace camera

#endif  // ENABLE_REALSENSE
