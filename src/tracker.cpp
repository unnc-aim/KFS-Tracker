#include "tracker.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <sstream>

namespace tracker {

QuadContourTracker::QuadContourTracker(float originWidth, float originHeight, const cv::Mat& cameraMatrix, int warpSize)
    : originWidth_(originWidth), originHeight_(originHeight), cameraMatrix_(cameraMatrix.clone()), warpSize_(warpSize) {}

DetectionResult QuadContourTracker::detect(const cv::Mat& input, const std::string& color) const {
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

    std::array<cv::Point2f, 4> bestCorners{};
    double bestArea = 0.0;

    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area < 100.0) {
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
            bestCorners = corners;
        }
    }

    if (bestArea <= 0.0) {
        result.annotated = input.clone();
        return result;
    }

    result.warped = warpToSquare(input, bestCorners);
    std::vector<cv::Point3f> objectPoints = buildObjectPoints();
    std::vector<cv::Point2f> imagePoints(bestCorners.begin(), bestCorners.end());

    cv::Vec3d rvec;
    cv::Vec3d tvec;
    bool pnpOk = cv::solvePnP(objectPoints, imagePoints, cameraMatrix_, cv::Mat(), rvec, tvec, false, cv::SOLVEPNP_IPPE);
    if (!pnpOk) {
        pnpOk = cv::solvePnP(objectPoints, imagePoints, cameraMatrix_, cv::Mat(), rvec, tvec);
    }

    result.found = true;
    result.corners = bestCorners;
    result.rvec = rvec;
    result.tvec = tvec;
    result.annotated = drawAnnotation(input, bestCorners, rvec, tvec);
    return result;
}

cv::Mat QuadContourTracker::buildColorMask(const cv::Mat& hsv, const std::string& color) const {
    std::vector<cv::Mat> hsvChannels;
    cv::split(hsv, hsvChannels);

    const cv::Mat& h = hsvChannels[0];
    const cv::Mat& s = hsvChannels[1];
    const cv::Mat& v = hsvChannels[2];

    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);

    std::vector<cv::Mat> bgrChannels;
    cv::split(bgr, bgrChannels);

    const cv::Mat& b = bgrChannels[0];
    const cv::Mat& g = bgrChannels[1];
    const cv::Mat& r = bgrChannels[2];

    cv::Mat mask1, mask2, baseMask, mask;
    cv::Mat hueMask, satMask, valueMask, washedOutMask, dominantMask, fallbackMask;

    if (color == "red") {
        cv::inRange(hsv, cv::Scalar(0, 60, 40), cv::Scalar(12, 255, 255), mask1);
        cv::inRange(hsv, cv::Scalar(168, 60, 40), cv::Scalar(180, 255, 255), mask2);
        cv::bitwise_or(mask1, mask2, baseMask);

        cv::inRange(h, 0, 18, mask1);
        cv::inRange(h, 162, 180, mask2);
        cv::bitwise_or(mask1, mask2, hueMask);

        cv::inRange(s, 12, 140, satMask);
        cv::inRange(v, 185, 255, valueMask);
        cv::bitwise_and(hueMask, satMask, washedOutMask);
        cv::bitwise_and(washedOutMask, valueMask, washedOutMask);

        cv::Mat gbMax;
        cv::max(g, b, gbMax);
        cv::compare(r, gbMax + 18, dominantMask, cv::CMP_GT);
        cv::inRange(r, 150, 255, mask1);
        cv::bitwise_and(dominantMask, mask1, dominantMask);
        cv::bitwise_and(dominantMask, valueMask, dominantMask);
    } else {
        cv::inRange(hsv, cv::Scalar(90, 60, 40), cv::Scalar(132, 255, 255), baseMask);

        cv::inRange(h, 88, 140, hueMask);
        cv::inRange(s, 10, 135, satMask);
        cv::inRange(v, 175, 255, valueMask);
        cv::bitwise_and(hueMask, satMask, washedOutMask);
        cv::bitwise_and(washedOutMask, valueMask, washedOutMask);

        cv::Mat rgMax;
        cv::max(r, g, rgMax);
        cv::compare(b, rgMax + 15, dominantMask, cv::CMP_GT);
        cv::inRange(b, 120, 255, mask1);
        cv::bitwise_and(dominantMask, mask1, dominantMask);
        cv::bitwise_and(dominantMask, valueMask, dominantMask);
    }

    cv::bitwise_or(washedOutMask, dominantMask, fallbackMask);
    cv::bitwise_or(baseMask, fallbackMask, mask);
    return mask;
}

std::array<cv::Point2f, 4> QuadContourTracker::orderCorners(const std::vector<cv::Point>& pts) {
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

bool QuadContourTracker::isRectangleLike(const std::array<cv::Point2f, 4>& corners) {
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

cv::Mat QuadContourTracker::warpToSquare(const cv::Mat& input, const std::array<cv::Point2f, 4>& corners) const {
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

std::vector<cv::Point3f> QuadContourTracker::buildObjectPoints() const {
    float w = originWidth_ * 0.5f;
    float h = originHeight_ * 0.5f;
    return {
        {-w, -h, 0.0f},
        {w, -h, 0.0f},
        {w, h, 0.0f},
        {-w, h, 0.0f}
    };
}

cv::Mat QuadContourTracker::drawAnnotation(const cv::Mat& input, const std::array<cv::Point2f, 4>& corners, const cv::Vec3d& rvec, const cv::Vec3d& tvec) const {
    cv::Mat annotated = input.clone();
    for (int i = 0; i < 4; ++i) {
        cv::line(annotated, corners[i], corners[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
        cv::circle(annotated, corners[i], 4, cv::Scalar(0, 0, 255), -1);
    }

    cv::putText(annotated, "rvec: [" + vecToString(rvec) + "]", cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);
    cv::putText(annotated, "tvec: [" + vecToString(tvec) + "]", cv::Point(20, 55), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);
    return annotated;
}

std::string QuadContourTracker::vecToString(const cv::Vec3d& v) {
    std::ostringstream oss;
    oss << v[0] << ", " << v[1] << ", " << v[2];
    return oss.str();
}

} // namespace tracker
