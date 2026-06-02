#include "2D_tracker.h"
#include "heatmap_classifier_infer.h"
#include <opencv2/opencv.hpp>

#include <iostream>
#include <memory>
#include <string>
#include <vector>

static std::string getArgValue(int argc, char** argv, const std::string& key, const std::string& def) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (key == argv[i]) {
            return argv[i + 1];
        }
    }
    return def;
}

int main(int argc, char** argv) {
    try {
        std::string imagePath = getArgValue(argc, argv, "--image", "image.jpg");
        std::string outputPath = getArgValue(argc, argv, "--output", "annotated.jpg");
        std::string heatmapModelPath = getArgValue(argc, argv, "--heatmap_model", "model/corner_heatmap_localizer.pt");
        std::string classifierModelPath = getArgValue(argc, argv, "--classifier_model", "model/tracker_classifier.pt");
        int inputSize = std::stoi(getArgValue(argc, argv, "--input_size", "300"));
        float originWidth = std::stof(getArgValue(argc, argv, "--origin_width", "0.35"));
        float originHeight = std::stof(getArgValue(argc, argv, "--origin_height", "0.35"));

        double fx = std::stod(getArgValue(argc, argv, "--fx", "0"));
        double fy = std::stod(getArgValue(argc, argv, "--fy", "0"));
        double cx = std::stod(getArgValue(argc, argv, "--cx", "0"));
        double cy = std::stod(getArgValue(argc, argv, "--cy", "0"));

        cv::Mat img = cv::imread(imagePath);
        if (img.empty()) {
            std::cerr << "Could not open image: " << imagePath << std::endl;
            return -1;
        }

        if (fx <= 0.0) fx = static_cast<double>(img.cols);
        if (fy <= 0.0) fy = static_cast<double>(img.rows);
        if (cx <= 0.0) cx = static_cast<double>(img.cols) * 0.5;
        if (cy <= 0.0) cy = static_cast<double>(img.rows) * 0.5;
        cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);

        tracker::HeatmapCornerInfer heatmapInfer(heatmapModelPath, inputSize);
        tracker::HeatmapInferResult heatmapResult = heatmapInfer.infer(img);

        if (!heatmapResult.valid) {
            std::cerr << "Heatmap tracker failed to produce corners." << std::endl;
            return -1;
        }

        tracker::PlaneTracker2D planeTracker(originWidth, originHeight, cameraMatrix);
        tracker::PlaneTracker2DResult poseResult = planeTracker.detectFromCorners(img, heatmapResult.corners);
        if (!poseResult.found) {
            std::cerr << "Pose estimation failed (likely reprojection error too large)." << std::endl;
            cv::imwrite(outputPath, heatmapResult.annotated.empty() ? img : heatmapResult.annotated);
            return 0;
        }

        tracker::ImageClassifierInfer classifierInfer(classifierModelPath, inputSize);
        tracker::ClassifierInferResult clsResult = classifierInfer.infer(img);

        cv::Mat finalAnnotated = poseResult.annotated.empty() ? img.clone() : poseResult.annotated.clone();
        if (clsResult.valid) {
            std::string clsText = "class: " + clsResult.className + " (id=" + std::to_string(clsResult.classId) +
                                  ", p=" + cv::format("%.3f", clsResult.classScore) + ")";
            cv::putText(finalAnnotated, clsText, cv::Point(20, 105), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 0), 2);
        }

        std::cout << "Detected corners (TL, BL, BR, TR):" << std::endl;
        for (size_t i = 0; i < heatmapResult.corners.size(); ++i) {
            std::cout << "  [" << i << "] " << heatmapResult.corners[i] << std::endl;
        }
        std::cout << "Pose rvec: " << poseResult.rvec << std::endl;
        std::cout << "Pose tvec: " << poseResult.tvec << std::endl;
        std::cout << "Pose reprojection error: " << poseResult.reprojectionError << std::endl;
        if (clsResult.valid) {
            std::cout << "Classification: " << clsResult.className << " (id=" << clsResult.classId
                      << ", score=" << clsResult.classScore << ")" << std::endl;
        }

        cv::imwrite(outputPath, finalAnnotated);
        std::cout << "Saved: " << outputPath << std::endl;
        return 0;
    } catch (const cv::Exception& e) {
        std::cerr << "[OpenCV][main] " << e.what() << std::endl;
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "[std::exception][main] " << e.what() << std::endl;
        return -1;
    }
}
