#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

struct DetectionResult {
    bool found = false;
    std::array<cv::Point2f, 4> corners{};
    cv::Mat warped;
    cv::Vec3d rvec;
    cv::Vec3d tvec;
    cv::Mat annotated;
};

class QuadContourDetector {
public:
    QuadContourDetector(float originWidth, float originHeight, const cv::Mat& cameraMatrix, int warpSize = 224)
        : originWidth_(originWidth), originHeight_(originHeight), cameraMatrix_(cameraMatrix.clone()), warpSize_(warpSize) {}

    DetectionResult detect(const cv::Mat& input, const std::string& color) const {
        DetectionResult result;
        if (input.empty()) {
            return result;
        }

        cv::Mat hsv;
        cv::cvtColor(input, hsv, cv::COLOR_BGR2HSV);

        cv::Mat mask = buildColorMask(hsv, color);
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        std::vector<cv::Point> bestContour;
        std::array<cv::Point2f, 4> bestCorners{};
        double bestArea = 0.0;

        for (const auto& contour : contours) {
            double area = cv::contourArea(contour);
            if (area < 1000.0) {
                continue;
            }

            std::vector<cv::Point> approx;
            double peri = cv::arcLength(contour, true);
            cv::approxPolyDP(contour, approx, 0.02 * peri, true);
            if (approx.size() != 4 || !cv::isContourConvex(approx)) {
                continue;
            }

            std::array<cv::Point2f, 4> corners = orderCorners(approx);
            if (!isRectangleLike(corners)) {
                continue;
            }

            if (area > bestArea) {
                bestArea = area;
                bestContour = contour;
                bestCorners = corners;
            }
        }

        if (bestArea <= 0.0) {
            result.annotated = input.clone();
            return result;
        }

        cv::Mat warped = warpToSquare(input, bestCorners);
        std::vector<cv::Point3f> objectPoints = buildObjectPoints();
        std::vector<cv::Point2f> imagePoints(bestCorners.begin(), bestCorners.end());

        cv::Vec3d rvec, tvec;
        bool pnpOk = cv::solvePnP(objectPoints, imagePoints, cameraMatrix_, cv::Mat(), rvec, tvec, false, cv::SOLVEPNP_IPPE);
        if (!pnpOk) {
            pnpOk = cv::solvePnP(objectPoints, imagePoints, cameraMatrix_, cv::Mat(), rvec, tvec);
        }

        result.found = true;
        result.corners = bestCorners;
        result.warped = warped;
        result.rvec = rvec;
        result.tvec = tvec;
        result.annotated = drawAnnotation(input, bestCorners, rvec, tvec);
        return result;
    }

private:
    cv::Mat buildColorMask(const cv::Mat& hsv, const std::string& color) const {
        cv::Mat mask1, mask2, mask;
        if (color == "red") {
            cv::inRange(hsv, cv::Scalar(0, 70, 50), cv::Scalar(10, 255, 255), mask1);
            cv::inRange(hsv, cv::Scalar(160, 70, 50), cv::Scalar(180, 255, 255), mask2);
            cv::bitwise_or(mask1, mask2, mask);
        } else {
            cv::inRange(hsv, cv::Scalar(90, 70, 50), cv::Scalar(130, 255, 255), mask);
        }
        return mask;
    }

    static std::array<cv::Point2f, 4> orderCorners(const std::vector<cv::Point>& pts) {
        std::array<cv::Point2f, 4> ordered{};
        std::vector<cv::Point2f> p;
        p.reserve(4);
        for (const auto& pt : pts) {
            p.emplace_back(static_cast<float>(pt.x), static_cast<float>(pt.y));
        }

        auto sumCmp = [](const cv::Point2f& a, const cv::Point2f& b) { return a.x + a.y < b.x + b.y; };
        auto diffCmp = [](const cv::Point2f& a, const cv::Point2f& b) { return a.x - a.y < b.x - b.y; };

        ordered[0] = *std::min_element(p.begin(), p.end(), sumCmp);
        ordered[2] = *std::max_element(p.begin(), p.end(), sumCmp);
        ordered[1] = *std::min_element(p.begin(), p.end(), diffCmp);
        ordered[3] = *std::max_element(p.begin(), p.end(), diffCmp);
        return ordered;
    }

    static bool isRectangleLike(const std::array<cv::Point2f, 4>& corners) {
        auto dist = [](const cv::Point2f& a, const cv::Point2f& b) {
            return std::hypot(a.x - b.x, a.y - b.y);
        };

        double w1 = dist(corners[0], corners[1]);
        double w2 = dist(corners[2], corners[3]);
        double h1 = dist(corners[0], corners[3]);
        double h2 = dist(corners[1], corners[2]);
        double w = (w1 + w2) * 0.5;
        double h = (h1 + h2) * 0.5;
        double ratio = (std::min)(w, h) / (std::max)(w, h);
        return ratio > 0.45;
    }

    cv::Mat warpToSquare(const cv::Mat& input, const std::array<cv::Point2f, 4>& corners) const {
        std::vector<cv::Point2f> src(corners.begin(), corners.end());
        std::vector<cv::Point2f> dst = {
            {0.0f, 0.0f},
            {static_cast<float>(warpSize_ - 1), 0.0f},
            {static_cast<float>(warpSize_ - 1), static_cast<float>(warpSize_ - 1)},
            {0.0f, static_cast<float>(warpSize_ - 1)}
        };

        cv::Mat m = cv::getPerspectiveTransform(src, dst);
        cv::Mat warped;
        cv::warpPerspective(input, warped, m, cv::Size(warpSize_, warpSize_));
        return warped;
    }

    std::vector<cv::Point3f> buildObjectPoints() const {
        float w = originWidth_ * 0.5f;
        float h = originHeight_ * 0.5f;
        return {
            {-w, -h, 0.0f},
            {w, -h, 0.0f},
            {w, h, 0.0f},
            {-w, h, 0.0f}
        };
    }

    cv::Mat drawAnnotation(const cv::Mat& input, const std::array<cv::Point2f, 4>& corners, const cv::Vec3d& rvec, const cv::Vec3d& tvec) const {
        cv::Mat annotated = input.clone();
        for (int i = 0; i < 4; ++i) {
            cv::line(annotated, corners[i], corners[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
            cv::circle(annotated, corners[i], 4, cv::Scalar(0, 0, 255), -1);
        }

        cv::putText(annotated, "rvec: [" + vecToString(rvec) + "]", cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);
        cv::putText(annotated, "tvec: [" + vecToString(tvec) + "]", cv::Point(20, 55), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);
        return annotated;
    }

    static std::string vecToString(const cv::Vec3d& v) {
        std::ostringstream oss;
        oss << v[0] << ", " << v[1] << ", " << v[2];
        return oss.str();
    }

    float originWidth_;
    float originHeight_;
    cv::Mat cameraMatrix_;
    int warpSize_;
};

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

    float originWidth = std::stof(getArgValue(argc, argv, "--origin_width", "0.20"));
    float originHeight = std::stof(getArgValue(argc, argv, "--origin_height", "0.12"));

    cv::Mat img = cv::imread(imagePath);
    if (img.empty()) {
        std::cerr << "Could not open image: " << imagePath << std::endl;
        return -1;
    }

    cv::Mat cameraMatrix = buildCameraMatrixFromArgs(argc, argv, img.size());
    QuadContourDetector detector(originWidth, originHeight, cameraMatrix);
    DetectionResult result = detector.detect(img, color);

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
