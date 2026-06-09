#pragma once

#include <opencv2/opencv.hpp>

#include <memory>
#include <string>

namespace camera {

// 相机内参：3x3 相机矩阵 K + 畸变系数
struct Intrinsics {
    cv::Mat cameraMatrix;   // 3x3 CV_64F，相机内参矩阵 K = [fx 0 cx; 0 fy cy; 0 0 1]
    cv::Mat distCoeffs;     // 1xN CV_64F，畸变系数（典型 5/8/14 个）
};

// 每帧的数据包：彩色图 + 深度图（可选）+ 该帧对应的内参
struct FrameBundle {
    cv::Mat color;          // BGR 彩色帧
    cv::Mat depth;          // 16UC1 深度图，单位 mm（仅 RealSense 提供，其他源为空）
    Intrinsics intrinsics;  // 该流的内参（可能在运行中变化，例如 RealSense 切分辨率）
};

// 相机源抽象基类
class CameraProvider {
public:
    virtual ~CameraProvider() = default;

    // 抓取下一帧。返回 false 表示流结束或出错。
    virtual bool grab(FrameBundle& bundle) = 0;

    virtual int frameWidth() const = 0;
    virtual int frameHeight() const = 0;
    virtual std::string providerName() const = 0;
};

// 工厂创建参数
struct CameraConfig {
    std::string type = "opencv";     // "opencv" | "realsense" | "file"
    std::string intrinsicsPath;      // 内参 YAML 路径（空 = 用 identity 内参 + 警告）
    int cameraIndex = 0;             // opencv 模式下的摄像头索引
    std::string videoPath;           // file 模式下的视频源（视频文件 / URL / 设备路径）
    int width = 1280;                // 期望采集宽度
    int height = 720;                // 期望采集高度
};

// 工厂函数：根据 cfg.type 构造对应的 provider。
// 失败时返回 nullptr 并向 stderr 打印错误。
std::unique_ptr<CameraProvider> createCameraProvider(const CameraConfig& cfg);

// 从 YAML 文件加载内参（OpenCV 标准标定格式）。
// 文件格式异常时抛 std::runtime_error。
Intrinsics loadIntrinsicsFromFile(const std::string& path);

// 根据给定分辨率生成默认 identity 内参（用于无标定时的兜底，不可信赖）。
Intrinsics makeDefaultIntrinsics(int width, int height);

}  // namespace camera
