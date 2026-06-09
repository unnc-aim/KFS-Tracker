#include "camera_provider.h"

#include <iostream>
#include <stdexcept>

namespace camera {

// 从 OpenCV 标准标定 YAML 文件加载内参。
// 期望字段：camera_matrix (3x3) + distortion_coefficients (1xN)
Intrinsics loadIntrinsicsFromFile(const std::string& path) {
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        throw std::runtime_error("Cannot open intrinsics file: " + path);
    }

    Intrinsics result;
    fs["camera_matrix"] >> result.cameraMatrix;
    fs["distortion_coefficients"] >> result.distCoeffs;

    if (result.cameraMatrix.empty()) {
        throw std::runtime_error("Missing 'camera_matrix' in intrinsics file: " + path);
    }
    // 畸变系数缺失时按 0 处理（标定结果可能不含畸变）
    if (result.distCoeffs.empty()) {
        result.distCoeffs = cv::Mat::zeros(1, 5, CV_64F);
    }

    // 强制统一为 CV_64F，避免下游因数据类型不一致出问题
    result.cameraMatrix.convertTo(result.cameraMatrix, CV_64F);
    result.distCoeffs.convertTo(result.distCoeffs, CV_64F);
    return result;
}

// 兜底 identity 内参：焦距近似为图像宽度，光心在图像中心。
// 仅用于让下游代码能跑起来，不能用于真实位姿估计。
Intrinsics makeDefaultIntrinsics(int width, int height) {
    Intrinsics intr;
    const double fx = static_cast<double>(width);
    const double fy = static_cast<double>(width);
    const double cx = static_cast<double>(width) * 0.5;
    const double cy = static_cast<double>(height) * 0.5;

    intr.cameraMatrix = (cv::Mat_<double>(3, 3) <<
        fx, 0.0, cx,
        0.0, fy, cy,
        0.0, 0.0, 1.0);
    intr.distCoeffs = cv::Mat::zeros(1, 5, CV_64F);
    return intr;
}

// 前向声明：具体实现分别在 camera_provider_opencv.cpp 与
// camera_provider_realsense.cpp 中（后者仅条件编译）
std::unique_ptr<CameraProvider> makeOpenCVProvider(const CameraConfig& cfg);
std::unique_ptr<CameraProvider> makeFileProvider(const CameraConfig& cfg);
#ifdef ENABLE_REALSENSE
std::unique_ptr<CameraProvider> makeRealSenseProvider(const CameraConfig& cfg);
#endif

// 工厂函数：根据 cfg.type 分发到对应实现
std::unique_ptr<CameraProvider> createCameraProvider(const CameraConfig& cfg) {
    if (cfg.type == "opencv") {
        return makeOpenCVProvider(cfg);
    }
    if (cfg.type == "file") {
        if (cfg.intrinsicsPath.empty()) {
            std::cerr << "Error: --intrinsics <path> is required for --camera_type file" << std::endl;
            return nullptr;
        }
        return makeFileProvider(cfg);
    }
    if (cfg.type == "realsense") {
#ifdef ENABLE_REALSENSE
        return makeRealSenseProvider(cfg);
#else
        std::cerr << "Error: RealSense support not compiled. "
                     "Rebuild with -DENABLE_REALSENSE=ON" << std::endl;
        return nullptr;
#endif
    }

    std::cerr << "Error: unknown camera type '" << cfg.type
              << "' (expected: opencv | realsense | file)" << std::endl;
    return nullptr;
}

}  // namespace camera
