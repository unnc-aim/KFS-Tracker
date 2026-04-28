#include "bbox_tracking_classify.h"
#include <opencv2/opencv.hpp>

#include <iostream>
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
        std::string modelPath = getArgValue(argc, argv, "--model", "model/tracker_localizer.pt");
        int inputSize = std::stoi(getArgValue(argc, argv, "--input_size", "300"));

        cv::Mat img = cv::imread(imagePath);
        if (img.empty()) {
            std::cerr << "Could not open image: " << imagePath << std::endl;
            return -1;
        }

        tracker::BboxTrackingClassifier predictor(modelPath, inputSize);
        tracker::BboxClassificationResult result = predictor.infer(img);

        if (!result.valid) {
            std::cout << "Inference finished, but bbox is invalid/empty." << std::endl;
        } else {
            std::cout << "Bbox contour points:" << std::endl;
            for (size_t i = 0; i < result.bboxContour.size(); ++i) {
                std::cout << "  [" << i << "] " << result.bboxContour[i] << std::endl;
            }
        }
        std::cout << "Classification: " << result.className << " (id=" << result.classId
                  << ", score=" << result.classScore << ")" << std::endl;

        cv::imwrite(outputPath, result.annotated);
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
