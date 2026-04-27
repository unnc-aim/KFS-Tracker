#include "tracker.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>

namespace tracker {

namespace {

cv::Point2f averagePoint(const std::vector<cv::Point2f>& points) {
    cv::Point2f center(0.0f, 0.0f);
    if (points.empty()) {
        return center;
    }

    for (const auto& point : points) {
        center += point;
    }
    center.x /= static_cast<float>(points.size());
    center.y /= static_cast<float>(points.size());
    return center;
}

}  // namespace

CubeTracker3D::CubeTracker3D(float originWidth,
                             float originHeight,
                             float originDepth,
                             const cv::Mat& cameraMatrix,
                             const cv::Mat& distCoeffs,
                             int warpSize)
    : originWidth_(originWidth),
      originHeight_(originHeight),
      originDepth_(originDepth),
      cameraMatrix_(cameraMatrix.clone()),
      distCoeffs_(distCoeffs.empty() ? cv::Mat::zeros(1, 5, CV_64F) : distCoeffs.clone()),
      warpSize_(warpSize) {}

PoseEstimationResult CubeTracker3D::detect(const cv::Mat& input, const std::string& color) const {
    PoseEstimationResult result;
    if (input.empty()) {
        return result;
    }

    cv::Mat hsv;
    cv::cvtColor(input, hsv, cv::COLOR_BGR2HSV);
    result.mask = buildColorMask(hsv, color);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(result.mask, result.mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(result.mask, result.mask, cv::MORPH_CLOSE, kernel);
    cv::dilate(result.mask, result.mask, kernel, cv::Point(-1, -1), 1);

    std::array<cv::Point2f, 4> faceCorners = extractVisibleFace(result.mask);
    if (cv::norm(faceCorners[0] - faceCorners[2]) < 1.0f) {
        result.annotated = input.clone();
        return result;
    }

    result.faceCorners = faceCorners;
    result.warpedFace = warpFaceToSquare(input, faceCorners);

    std::vector<cv::Point3f> frontFaceObjectPoints = buildFrontFaceObjectPoints();
    std::vector<cv::Point2f> imageFacePoints(faceCorners.begin(), faceCorners.end());

    cv::Vec3d rvec;
    cv::Vec3d tvec;
    bool poseOk = cv::solvePnP(frontFaceObjectPoints,
                               imageFacePoints,
                               cameraMatrix_,
                               distCoeffs_,
                               rvec,
                               tvec,
                               false,
                               cv::SOLVEPNP_IPPE);
    if (!poseOk) {
        poseOk = cv::solvePnPRansac(frontFaceObjectPoints,
                                    imageFacePoints,
                                    cameraMatrix_,
                                    distCoeffs_,
                                    rvec,
                                    tvec,
                                    false,
                                    100,
                                    8.0f,
                                    0.99);
    }

    if (!poseOk) {
        result.annotated = input.clone();
        return result;
    }

    std::vector<cv::Point3f> cubeObjectPoints = buildCubeObjectPoints();
    std::vector<cv::Point2f> projectedPoints;
    cv::projectPoints(cubeObjectPoints, rvec, tvec, cameraMatrix_, distCoeffs_, projectedPoints);

    cv::Mat rotationMatrix;
    cv::Rodrigues(rvec, rotationMatrix);

    std::vector<cv::Point3f> visibleObjectPoints;
    std::vector<cv::Point2f> visibleImagePoints = computeVisibleHullPoints(projectedPoints, cubeObjectPoints, &visibleObjectPoints);
    std::vector<cv::Point3f> visibleCameraPoints = transformToCameraCoordinates(visibleObjectPoints, rotationMatrix, tvec);

    result.found = !visibleImagePoints.empty();
    result.imageCorners2d = std::move(visibleImagePoints);
    result.objectCorners3d = std::move(visibleObjectPoints);
    result.cameraCorners3d = std::move(visibleCameraPoints);
    result.rvec = rvec;
    result.tvec = tvec;
    result.rotationMatrix = rotationMatrix;
    result.reprojectionError = computeReprojectionError(frontFaceObjectPoints, imageFacePoints, rvec, tvec);
    result.annotated = drawAnnotation(input, faceCorners, projectedPoints, rvec, tvec, result.reprojectionError);
    return result;
}

cv::Mat CubeTracker3D::buildColorMask(const cv::Mat& hsv, const std::string& color) const {
    cv::Mat mask1;
    cv::Mat mask2;
    cv::Mat baseMask;
    cv::Mat saturationMask;
    cv::Mat valueMask;
    cv::Mat mask;

    if (color == "red") {
        cv::inRange(hsv, cv::Scalar(0, 70, 45), cv::Scalar(12, 255, 255), mask1);
        cv::inRange(hsv, cv::Scalar(165, 70, 45), cv::Scalar(180, 255, 255), mask2);
        cv::bitwise_or(mask1, mask2, baseMask);
    } else {
        cv::inRange(hsv, cv::Scalar(90, 70, 45), cv::Scalar(135, 255, 255), baseMask);
    }

    cv::inRange(hsv, cv::Scalar(0, 40, 40), cv::Scalar(180, 255, 255), saturationMask);
    cv::inRange(hsv, cv::Scalar(0, 0, 50), cv::Scalar(180, 255, 255), valueMask);
    cv::bitwise_and(baseMask, saturationMask, mask);
    cv::bitwise_and(mask, valueMask, mask);
    return mask;
}

std::array<cv::Point2f, 4> CubeTracker3D::extractVisibleFace(const cv::Mat& mask) const {
    std::array<cv::Point2f, 4> bestCorners{};
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    double bestArea = 0.0;
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area < 500.0) {
            continue;
        }

        std::vector<cv::Point> approx;
        double perimeter = cv::arcLength(contour, true);
        cv::approxPolyDP(contour, approx, 0.02 * perimeter, true);

        std::array<cv::Point2f, 4> candidate{};
        bool valid = false;
        if (approx.size() == 4 && cv::isContourConvex(approx)) {
            candidate = orderCorners(approx);
            valid = isRectangleLike(candidate);
        } else {
            cv::RotatedRect box = cv::minAreaRect(contour);
            cv::Point2f boxPoints[4];
            box.points(boxPoints);
            std::vector<cv::Point> boxContour(4);
            for (int i = 0; i < 4; ++i) {
                boxContour[i] = boxPoints[i];
            }
            candidate = orderCorners(boxContour);
            valid = isRectangleLike(candidate);
        }

        if (valid && area > bestArea) {
            bestArea = area;
            bestCorners = candidate;
        }
    }

    return bestCorners;
}

std::array<cv::Point2f, 4> CubeTracker3D::orderCorners(const std::vector<cv::Point>& pts) {
    std::array<cv::Point2f, 4> ordered{};
    std::vector<cv::Point2f> points;
    points.reserve(pts.size());
    for (const auto& pt : pts) {
        points.emplace_back(static_cast<float>(pt.x), static_cast<float>(pt.y));
    }

    auto sumCmp = [](const cv::Point2f& a, const cv::Point2f& b) { return a.x + a.y < b.x + b.y; };
    auto diffCmp = [](const cv::Point2f& a, const cv::Point2f& b) { return a.x - a.y < b.x - b.y; };

    ordered[0] = *std::min_element(points.begin(), points.end(), sumCmp);
    ordered[2] = *std::max_element(points.begin(), points.end(), sumCmp);
    ordered[1] = *std::min_element(points.begin(), points.end(), diffCmp);
    ordered[3] = *std::max_element(points.begin(), points.end(), diffCmp);
    return ordered;
}

bool CubeTracker3D::isRectangleLike(const std::array<cv::Point2f, 4>& corners) {
    auto length = [](const cv::Point2f& a, const cv::Point2f& b) {
        return std::hypot(a.x - b.x, a.y - b.y);
    };

    double widthA = length(corners[0], corners[1]);
    double widthB = length(corners[3], corners[2]);
    double heightA = length(corners[0], corners[3]);
    double heightB = length(corners[1], corners[2]);
    double shortSide = (std::min)((widthA + widthB) * 0.5, (heightA + heightB) * 0.5);
    double longSide = (std::max)((widthA + widthB) * 0.5, (heightA + heightB) * 0.5);
    return longSide > 1.0 && (shortSide / longSide) > 0.45;
}

cv::Mat CubeTracker3D::warpFaceToSquare(const cv::Mat& input, const std::array<cv::Point2f, 4>& corners) const {
    std::vector<cv::Point2f> src(corners.begin(), corners.end());
    std::vector<cv::Point2f> dst = {
        {0.0f, 0.0f},
        {static_cast<float>(warpSize_ - 1), 0.0f},
        {static_cast<float>(warpSize_ - 1), static_cast<float>(warpSize_ - 1)},
        {0.0f, static_cast<float>(warpSize_ - 1)}
    };

    cv::Mat transform = cv::getPerspectiveTransform(src, dst);
    cv::Mat warped;
    cv::warpPerspective(input, warped, transform, cv::Size(warpSize_, warpSize_));
    return warped;
}

std::vector<cv::Point3f> CubeTracker3D::buildFrontFaceObjectPoints() const {
    float halfWidth = originWidth_ * 0.5f;
    float halfHeight = originHeight_ * 0.5f;
    float halfDepth = originDepth_ * 0.5f;

    return {
        {-halfWidth, -halfHeight, halfDepth},
        {halfWidth, -halfHeight, halfDepth},
        {halfWidth, halfHeight, halfDepth},
        {-halfWidth, halfHeight, halfDepth}
    };
}

std::vector<cv::Point3f> CubeTracker3D::buildCubeObjectPoints() const {
    float halfWidth = originWidth_ * 0.5f;
    float halfHeight = originHeight_ * 0.5f;
    float halfDepth = originDepth_ * 0.5f;

    return {
        {-halfWidth, -halfHeight, halfDepth},
        {halfWidth, -halfHeight, halfDepth},
        {halfWidth, halfHeight, halfDepth},
        {-halfWidth, halfHeight, halfDepth},
        {-halfWidth, -halfHeight, -halfDepth},
        {halfWidth, -halfHeight, -halfDepth},
        {halfWidth, halfHeight, -halfDepth},
        {-halfWidth, halfHeight, -halfDepth}
    };
}

std::vector<cv::Point3f> CubeTracker3D::transformToCameraCoordinates(const std::vector<cv::Point3f>& objectPoints,
                                                                     const cv::Mat& rotationMatrix,
                                                                     const cv::Vec3d& tvec) const {
    std::vector<cv::Point3f> cameraPoints;
    cameraPoints.reserve(objectPoints.size());

    for (const auto& objectPoint : objectPoints) {
        cv::Mat objectVector = (cv::Mat_<double>(3, 1) << objectPoint.x, objectPoint.y, objectPoint.z);
        cv::Mat cameraVector = rotationMatrix * objectVector + (cv::Mat_<double>(3, 1) << tvec[0], tvec[1], tvec[2]);
        cameraPoints.emplace_back(static_cast<float>(cameraVector.at<double>(0)),
                                  static_cast<float>(cameraVector.at<double>(1)),
                                  static_cast<float>(cameraVector.at<double>(2)));
    }

    return cameraPoints;
}

std::vector<cv::Point2f> CubeTracker3D::computeVisibleHullPoints(const std::vector<cv::Point2f>& projectedPoints,
                                                                 const std::vector<cv::Point3f>& objectPoints,
                                                                 std::vector<cv::Point3f>* visibleObjectPoints) const {
    std::vector<cv::Point2f> hullPoints;
    if (projectedPoints.size() < 4 || projectedPoints.size() != objectPoints.size()) {
        return hullPoints;
    }

    std::vector<int> hullIndices;
    cv::convexHull(projectedPoints, hullIndices, false, false);
    if (hullIndices.size() < 3) {
        return hullPoints;
    }

    cv::Point2f center = averagePoint(projectedPoints);
    std::vector<int> orderedIndices = hullIndices;
    std::sort(orderedIndices.begin(),
              orderedIndices.end(),
              [&projectedPoints, center](int lhs, int rhs) {
                  double angleL = std::atan2(projectedPoints[lhs].y - center.y, projectedPoints[lhs].x - center.x);
                  double angleR = std::atan2(projectedPoints[rhs].y - center.y, projectedPoints[rhs].x - center.x);
                  return angleL < angleR;
              });

    hullPoints.reserve(orderedIndices.size());
    if (visibleObjectPoints != nullptr) {
        visibleObjectPoints->clear();
        visibleObjectPoints->reserve(orderedIndices.size());
    }

    for (int index : orderedIndices) {
        hullPoints.push_back(projectedPoints[index]);
        if (visibleObjectPoints != nullptr) {
            visibleObjectPoints->push_back(objectPoints[index]);
        }
    }

    return hullPoints;
}

double CubeTracker3D::computeReprojectionError(const std::vector<cv::Point3f>& objectPoints,
                                               const std::vector<cv::Point2f>& imagePoints,
                                               const cv::Vec3d& rvec,
                                               const cv::Vec3d& tvec) const {
    if (objectPoints.empty() || objectPoints.size() != imagePoints.size()) {
        return -1.0;
    }

    std::vector<cv::Point2f> reprojectedPoints;
    cv::projectPoints(objectPoints, rvec, tvec, cameraMatrix_, distCoeffs_, reprojectedPoints);

    double errorSum = 0.0;
    for (size_t i = 0; i < imagePoints.size(); ++i) {
        errorSum += cv::norm(imagePoints[i] - reprojectedPoints[i]);
    }
    return errorSum / static_cast<double>(imagePoints.size());
}

cv::Mat CubeTracker3D::drawAnnotation(const cv::Mat& input,
                                      const std::array<cv::Point2f, 4>& faceCorners,
                                      const std::vector<cv::Point2f>& projectedCorners,
                                      const cv::Vec3d& rvec,
                                      const cv::Vec3d& tvec,
                                      double reprojectionError) const {
    cv::Mat annotated = input.clone();

    for (int i = 0; i < 4; ++i) {
        cv::line(annotated, faceCorners[i], faceCorners[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
        cv::circle(annotated, faceCorners[i], 4, cv::Scalar(0, 255, 255), -1);
    }

    if (projectedCorners.size() >= 3) {
        std::vector<cv::Point> polygon;
        polygon.reserve(projectedCorners.size());
        for (const auto& point : projectedCorners) {
            polygon.emplace_back(cvRound(point.x), cvRound(point.y));
            cv::circle(annotated, point, 5, cv::Scalar(255, 0, 255), -1);
        }
        cv::polylines(annotated, polygon, true, cv::Scalar(255, 0, 0), 2);
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
    cv::putText(annotated,
                "reproj err: " + cv::format("%.3f px", reprojectionError),
                cv::Point(20, 80),
                cv::FONT_HERSHEY_SIMPLEX,
                0.55,
                cv::Scalar(255, 255, 0),
                2);
    return annotated;
}

std::string CubeTracker3D::vecToString(const cv::Vec3d& v) {
    std::ostringstream oss;
    oss << v[0] << ", " << v[1] << ", " << v[2];
    return oss.str();
}

}  // namespace tracker
