#include "2D_tracker.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <vector>

namespace tracker {

namespace {

constexpr double kMinContourArea = 500.0;

double pointDistance(const cv::Point2f& a, const cv::Point2f& b) {
    return cv::norm(a - b);
}

}  // namespace

PlaneTracker2D::PlaneTracker2D(float originWidth,
                               float originHeight,
                               const cv::Mat& cameraMatrix,
                               const cv::Mat& distCoeffs,
                               int warpSize)
    : originWidth_(originWidth),
      originHeight_(originHeight),
      cameraMatrix_(cameraMatrix.clone()),
      distCoeffs_(distCoeffs.empty() ? cv::Mat::zeros(1, 5, CV_64F) : distCoeffs.clone()),
      warpSize_(warpSize) {}

PlaneTracker2DResult PlaneTracker2D::detect(const cv::Mat& input, const std::string& color) const {
    PlaneTracker2DResult result;
    if (input.empty()) {
        return result;
    }

    try {
        result.mask = buildColorMask(input, color);
        refineMask(result.mask);

        QuadCandidate candidate = extractLargestQuad(result.mask);
        if (!candidate.valid) {
            result.annotated = input.clone();
            return result;
        }

        cv::Mat homography;
        result.warped = warpToSquare(input, candidate.corners, &homography);
        result.homography = homography;
        result.corners = candidate.corners;
        result.contourArea = candidate.area;

        std::vector<cv::Point3f> objectPoints = buildObjectPoints();
        std::vector<cv::Point2f> imagePoints(candidate.corners.begin(), candidate.corners.end());

        bool pnpOk = cv::solvePnP(objectPoints,
                                  imagePoints,
                                  cameraMatrix_,
                                  distCoeffs_,
                                  result.rvec,
                                  result.tvec,
                                  false,
                                  cv::SOLVEPNP_IPPE);
        if (!pnpOk) {
            pnpOk = cv::solvePnP(objectPoints,
                                 imagePoints,
                                 cameraMatrix_,
                                 distCoeffs_,
                                 result.rvec,
                                 result.tvec);
        }

        if (!pnpOk) {
            result.annotated = drawAnnotation(input, candidate.contour, candidate.corners, cv::Vec3d(), cv::Vec3d(), -1.0);
            return result;
        }

        result.reprojectionError = computeReprojectionError(objectPoints, imagePoints, result.rvec, result.tvec);
        if (!std::isfinite(result.reprojectionError) || result.reprojectionError > maxReprojectionErrorPx_) {
            result.annotated = drawAnnotation(input, candidate.contour, candidate.corners, cv::Vec3d(), cv::Vec3d(), -1.0);
            result.reprojectionError = -1.0;
            return result;
        }
        result.annotated = drawAnnotation(input,
                                          candidate.contour,
                                          candidate.corners,
                                          result.rvec,
                                          result.tvec,
                                          result.reprojectionError);
        result.found = true;
        return result;
    } catch (const cv::Exception&) {
        result.annotated = input.clone();
        return result;
    }
}

PlaneTracker2DResult PlaneTracker2D::detectFromCorners(const cv::Mat& input, const std::array<cv::Point2f, 4>& corners) const {
    PlaneTracker2DResult result;
    if (input.empty()) {
        return result;
    }
    try {
        if (!isRectangleLike(corners)) {
            result.annotated = input.clone();
            return result;
        }

        result.corners = corners;
        result.contourArea = std::abs(cv::contourArea(std::vector<cv::Point2f>(corners.begin(), corners.end())));
        result.warped = warpToSquare(input, corners, &result.homography);

        std::vector<cv::Point3f> objectPoints = buildObjectPoints();
        std::vector<cv::Point2f> imagePoints(corners.begin(), corners.end());

        bool pnpOk = cv::solvePnP(objectPoints,
                                  imagePoints,
                                  cameraMatrix_,
                                  distCoeffs_,
                                  result.rvec,
                                  result.tvec,
                                  false,
                                  cv::SOLVEPNP_IPPE);
        if (!pnpOk) {
            pnpOk = cv::solvePnP(objectPoints, imagePoints, cameraMatrix_, distCoeffs_, result.rvec, result.tvec);
        }
        if (!pnpOk) {
            result.annotated = drawAnnotation(input, {}, corners, cv::Vec3d(), cv::Vec3d(), -1.0);
            return result;
        }

        result.reprojectionError = computeReprojectionError(objectPoints, imagePoints, result.rvec, result.tvec);
        if (!std::isfinite(result.reprojectionError) || result.reprojectionError > maxReprojectionErrorPx_) {
            result.annotated = drawAnnotation(input, {}, corners, cv::Vec3d(), cv::Vec3d(), -1.0);
            result.reprojectionError = -1.0;
            return result;
        }

        result.annotated = drawAnnotation(input, {}, corners, result.rvec, result.tvec, result.reprojectionError);
        result.found = true;
        return result;
    } catch (const cv::Exception&) {
        result.annotated = input.clone();
        return result;
    }
}

cv::Mat PlaneTracker2D::buildColorMask(const cv::Mat& input, const std::string& color) const {
    cv::Mat hsv;
    cv::cvtColor(input, hsv, cv::COLOR_BGR2HSV);

    cv::Mat mask;
    if (color == "red") {
        cv::Mat lowerMask;
        cv::Mat upperMask;
        cv::inRange(hsv, cv::Scalar(0, 70, 50), cv::Scalar(10, 255, 255), lowerMask);
        cv::inRange(hsv, cv::Scalar(160, 70, 50), cv::Scalar(180, 255, 255), upperMask);
        cv::bitwise_or(lowerMask, upperMask, mask);
    } else {
        cv::inRange(hsv, cv::Scalar(90, 70, 50), cv::Scalar(130, 255, 255), mask);
    }
    return mask;
}

void PlaneTracker2D::refineMask(cv::Mat& mask) {
    if (mask.empty()) {
        return;
    }

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    cv::GaussianBlur(mask, mask, cv::Size(5, 5), 0.0);
    cv::threshold(mask, mask, 127, 255, cv::THRESH_BINARY);
}

PlaneTracker2D::QuadCandidate PlaneTracker2D::extractLargestQuad(const cv::Mat& mask) const {
    QuadCandidate best;

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area < kMinContourArea) {
            continue;
        }

        std::vector<cv::Point> approx;
        double perimeter = cv::arcLength(contour, true);
        cv::approxPolyDP(contour, approx, 0.02 * perimeter, true);
        if (approx.size() != 4 || !cv::isContourConvex(approx)) {
            continue;
        }

        std::array<cv::Point2f, 4> corners = orderCorners(approx);
        if (!isRectangleLike(corners)) {
            continue;
        }

        if (!best.valid || area > best.area) {
            best.valid = true;
            best.area = area;
            best.contour = contour;
            best.corners = corners;
        }
    }

    return best;
}

std::array<cv::Point2f, 4> PlaneTracker2D::orderCorners(const std::vector<cv::Point>& polygon) {
    std::array<cv::Point2f, 4> ordered{};
    std::vector<cv::Point2f> points;
    points.reserve(polygon.size());
    for (const auto& point : polygon) {
        points.emplace_back(static_cast<float>(point.x), static_cast<float>(point.y));
    }

    auto sumComparator = [](const cv::Point2f& lhs, const cv::Point2f& rhs) {
        return lhs.x + lhs.y < rhs.x + rhs.y;
    };
    auto diffComparator = [](const cv::Point2f& lhs, const cv::Point2f& rhs) {
        return lhs.x - lhs.y < rhs.x - rhs.y;
    };

    ordered[0] = *std::min_element(points.begin(), points.end(), sumComparator);
    ordered[2] = *std::max_element(points.begin(), points.end(), sumComparator);
    ordered[1] = *std::min_element(points.begin(), points.end(), diffComparator);
    ordered[3] = *std::max_element(points.begin(), points.end(), diffComparator);
    return ordered;
}

bool PlaneTracker2D::isRectangleLike(const std::array<cv::Point2f, 4>& corners) {
    double w1 = pointDistance(corners[0], corners[1]);
    double w2 = pointDistance(corners[2], corners[3]);
    double h1 = pointDistance(corners[0], corners[3]);
    double h2 = pointDistance(corners[1], corners[2]);
    double width = (w1 + w2) * 0.5;
    double height = (h1 + h2) * 0.5;
    if (width < 1e-6 || height < 1e-6) {
        return false;
    }

    double ratio = (std::min)(width, height) / (std::max)(width, height);
    if (ratio < 0.45) {
        return false;
    }

    auto dotAngle = [](const cv::Point2f& prev, const cv::Point2f& current, const cv::Point2f& next) {
        cv::Point2f v1 = prev - current;
        cv::Point2f v2 = next - current;
        double denom = cv::norm(v1) * cv::norm(v2);
        if (denom < 1e-6) {
            return 1.0;
        }
        return (v1.x * v2.x + v1.y * v2.y) / denom;
    };

    for (int i = 0; i < 4; ++i) {
        const cv::Point2f& prev = corners[(i + 3) % 4];
        const cv::Point2f& current = corners[i];
        const cv::Point2f& next = corners[(i + 1) % 4];
        double cosValue = std::abs(dotAngle(prev, current, next));
        if (cosValue > 0.35) {
            return false;
        }
    }
    return true;
}

cv::Mat PlaneTracker2D::warpToSquare(const cv::Mat& input,
                                     const std::array<cv::Point2f, 4>& corners,
                                     cv::Mat* homography) const {
    std::vector<cv::Point2f> src(corners.begin(), corners.end());
    std::vector<cv::Point2f> dst = {
        {0.0f, 0.0f},
        {static_cast<float>(warpSize_ - 1), 0.0f},
        {static_cast<float>(warpSize_ - 1), static_cast<float>(warpSize_ - 1)},
        {0.0f, static_cast<float>(warpSize_ - 1)}
    };

    cv::Mat transform = cv::getPerspectiveTransform(src, dst);
    if (homography != nullptr) {
        *homography = transform.clone();
    }

    cv::Mat warped;
    cv::warpPerspective(input, warped, transform, cv::Size(warpSize_, warpSize_));
    return warped;
}

std::vector<cv::Point3f> PlaneTracker2D::buildObjectPoints() const {
    float halfWidth = originWidth_ * 0.5f;
    float halfHeight = originHeight_ * 0.5f;
    return {
        {-halfWidth, -halfHeight, 0.0f},
        {halfWidth, -halfHeight, 0.0f},
        {halfWidth, halfHeight, 0.0f},
        {-halfWidth, halfHeight, 0.0f}
    };
}

double PlaneTracker2D::computeReprojectionError(const std::vector<cv::Point3f>& objectPoints,
                                                const std::vector<cv::Point2f>& imagePoints,
                                                const cv::Vec3d& rvec,
                                                const cv::Vec3d& tvec) const {
    std::vector<cv::Point2f> reprojected;
    cv::projectPoints(objectPoints, rvec, tvec, cameraMatrix_, distCoeffs_, reprojected);
    if (reprojected.size() != imagePoints.size() || reprojected.empty()) {
        return -1.0;
    }

    double error = 0.0;
    for (size_t i = 0; i < reprojected.size(); ++i) {
        error += cv::norm(reprojected[i] - imagePoints[i]);
    }
    return error / static_cast<double>(reprojected.size());
}

cv::Mat PlaneTracker2D::drawAnnotation(const cv::Mat& input,
                                       const std::vector<cv::Point>& contour,
                                       const std::array<cv::Point2f, 4>& corners,
                                       const cv::Vec3d& rvec,
                                       const cv::Vec3d& tvec,
                                       double reprojectionError) const {
    cv::Mat annotated = input.clone();

    if (!contour.empty()) {
        std::vector<std::vector<cv::Point>> contours = {contour};
        cv::drawContours(annotated, contours, 0, cv::Scalar(255, 255, 0), 2);
    }

    for (int i = 0; i < 4; ++i) {
        cv::line(annotated, corners[i], corners[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
        cv::circle(annotated, corners[i], 4, cv::Scalar(0, 0, 255), -1);
    }

    cv::putText(annotated,
                "rvec: [" + vecToString(rvec) + "]",
                cv::Point(20, 30),
                cv::FONT_HERSHEY_SIMPLEX,
                0.55,
                cv::Scalar(255, 255, 0),
                2);
    cv::putText(annotated,
                "tvec: [" + vecToString(tvec) + "]",
                cv::Point(20, 55),
                cv::FONT_HERSHEY_SIMPLEX,
                0.55,
                cv::Scalar(255, 255, 0),
                2);
    if (reprojectionError >= 0.0) {
        cv::putText(annotated,
                    "reproj err: " + cv::format("%.3f", reprojectionError),
                    cv::Point(20, 80),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.55,
                    cv::Scalar(255, 255, 0),
                    2);
    }

    return annotated;
}

std::string PlaneTracker2D::vecToString(const cv::Vec3d& v) {
    std::ostringstream oss;
    oss << v[0] << ", " << v[1] << ", " << v[2];
    return oss.str();
}

}  // namespace tracker
