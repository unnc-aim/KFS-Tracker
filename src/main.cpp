#include "tracker.h"
#include <opencv2/opencv.hpp>

#include <iostream>
#include <memory>
#include <string>

static std::string getArgValue(int argc, char** argv, const std::string& key, const std::string& def) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (key == argv[i]) {
            return argv[i + 1];
        }
    }
    return def;
}

static cv::Mat buildCameraMatrixFromArgs(int argc, char** argv, const cv::Size& imageSize) {
    double fx = std::stod(getArgValue(argc, argv, "--fx", std::to_string(imageSize.width)));
    double fy = std::stod(getArgValue(argc, argv, "--fy", std::to_string(imageSize.height)));
    double cx = std::stod(getArgValue(argc, argv, "--cx", std::to_string(imageSize.width * 0.5)));
    double cy = std::stod(getArgValue(argc, argv, "--cy", std::to_string(imageSize.height * 0.5)));
    return (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
}

int main(int argc, char** argv) {
    std::string imagePath = getArgValue(argc, argv, "--image", "image.jpg");
    std::string color = getArgValue(argc, argv, "--color", "red");
    std::string outputPath = getArgValue(argc, argv, "--output", "annotated.jpg");

    float originWidth = std::stof(getArgValue(argc, argv, "--origin_width", "0.35"));
    float originHeight = std::stof(getArgValue(argc, argv, "--origin_height", "0.35"));

    cv::Mat img = cv::imread(imagePath);
    if (img.empty()) {
        std::cerr << "Could not open image: " << imagePath << std::endl;
        return -1;
    }

    cv::Mat cameraMatrix = buildCameraMatrixFromArgs(argc, argv, img.size());
    std::unique_ptr<tracker::TrackerInterface> tracker = std::make_unique<tracker::QuadContourTracker>(originWidth, originHeight, cameraMatrix);
    tracker::DetectionResult result = tracker->detect(img, color);

    if (!result.found) {
        std::cout << "No quadrilateral target found." << std::endl;
        return 0;
    }

    cv::imwrite(outputPath, result.annotated);
    cv::imwrite("warped.jpg", result.warped);

    std::cout << "Detected quad." << std::endl;
    std::cout << "rvec: " << result.rvec << std::endl;
    std::cout << "tvec: " << result.tvec << std::endl;
    std::cout << "Saved: " << outputPath << ", warped.jpg" << std::endl;
    return 0;
}
