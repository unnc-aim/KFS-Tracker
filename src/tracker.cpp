#include "tracker.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>

namespace tracker {

namespace {

constexpr double kMinContourArea = 500.0;
constexpr int kPolygonVertexCount = 6;

float wrapAngle(float angle) {
    while (angle < 0.0f) {
        angle += static_cast<float>(CV_PI);
    }
    while (angle >= static_cast<float>(CV_PI)) {
        angle -= static_cast<float>(CV_PI);
    }
    return angle;
}

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

double lineDistanceToPoint(const cv::Vec3f& line, const cv::Point2f& point) {
    return std::abs(line[0] * point.x + line[1] * point.y + line[2]) /
           std::sqrt(static_cast<double>(line[0] * line[0] + line[1] * line[1]) + 1e-9);
}

double polygonArea(const std::array<cv::Point2f, 6>& corners) {
    double area = 0.0;
    for (int i = 0; i < kPolygonVertexCount; ++i) {
        const auto& a = corners[i];
        const auto& b = corners[(i + 1) % kPolygonVertexCount];
        area += static_cast<double>(a.x) * b.y - static_cast<double>(b.x) * a.y;
    }
    return std::abs(area) * 0.5;
}

double distanceOrLarge(const cv::Point2f& point, const cv::Point2f& target) {
    if (std::isnan(point.x) || std::isnan(point.y) || std::isinf(point.x) || std::isinf(point.y) ||
        std::isnan(target.x) || std::isnan(target.y) || std::isinf(target.x) || std::isinf(target.y)) {
        return 5e3;
    }
    return cv::norm(point - target);
}

bool intersectLines(const cv::Vec3f& lhs, const cv::Vec3f& rhs, cv::Point2f* intersection) {
    cv::Vec3f cross = lhs.cross(rhs);
    if (std::abs(cross[2]) < 1e-6f) {
        return false;
    }

    intersection->x = cross[0] / cross[2];
    intersection->y = cross[1] / cross[2];
    return true;
}

std::vector<cv::Point2f> reducePolygonToSix(const std::vector<cv::Point>& polygon) {
    std::vector<cv::Point2f> reduced;
    if (polygon.size() < 6) {
        return reduced;
    }

    std::vector<cv::Point2f> points;
    points.reserve(polygon.size());
    for (const auto& point : polygon) {
        points.emplace_back(static_cast<float>(point.x), static_cast<float>(point.y));
    }

    cv::Point2f center = averagePoint(points);
    std::sort(points.begin(),
              points.end(),
              [center](const cv::Point2f& lhs, const cv::Point2f& rhs) {
                  return std::atan2(lhs.y - center.y, lhs.x - center.x) < std::atan2(rhs.y - center.y, rhs.x - center.x);
              });

    reduced.reserve(6);
    const float step = static_cast<float>(points.size()) / 6.0f;
    for (int i = 0; i < 6; ++i) {
        int index = std::min(static_cast<int>(std::round(i * step)), static_cast<int>(points.size()) - 1);
        reduced.push_back(points[index]);
    }
    return reduced;
}

std::array<cv::Point2f, 6> selectSixCornersFromContour(const std::vector<cv::Point>& contour) {
    std::array<cv::Point2f, 6> selected{};
    if (contour.size() < 6) {
        return selected;
    }

    std::vector<cv::Point2f> pts;
    pts.reserve(contour.size());
    for (const auto& p : contour) {
        pts.emplace_back(static_cast<float>(p.x), static_cast<float>(p.y));
    }

    cv::Point2f center = averagePoint(pts);
    std::vector<std::pair<float, int>> radial;
    radial.reserve(pts.size());
    for (int i = 0; i < static_cast<int>(pts.size()); ++i) {
        float r = cv::norm(pts[i] - center);
        radial.emplace_back(r, i);
    }
    std::sort(radial.begin(), radial.end(), [](const auto& lhs, const auto& rhs) { return lhs.first > rhs.first; });

    std::vector<cv::Point2f> candidates;
    candidates.reserve(12);
    for (int i = 0; i < static_cast<int>(radial.size()) && static_cast<int>(candidates.size()) < 12; ++i) {
        candidates.push_back(pts[radial[i].second]);
    }
    if (candidates.size() < 6) {
        return selected;
    }

    std::sort(candidates.begin(),
              candidates.end(),
              [center](const cv::Point2f& lhs, const cv::Point2f& rhs) {
                  return std::atan2(lhs.y - center.y, lhs.x - center.x) < std::atan2(rhs.y - center.y, rhs.x - center.x);
              });

    std::vector<cv::Point2f> deduped;
    deduped.reserve(candidates.size());
    for (const auto& c : candidates) {
        bool tooClose = false;
        for (const auto& d : deduped) {
            if (cv::norm(c - d) < 12.0f) {
                tooClose = true;
                break;
            }
        }
        if (!tooClose) {
            deduped.push_back(c);
        }
    }

    if (deduped.size() < 6) {
        return selected;
    }

    if (deduped.size() > 6) {
        std::vector<cv::Point2f> compact;
        compact.reserve(6);
        const float step = static_cast<float>(deduped.size()) / 6.0f;
        for (int i = 0; i < 6; ++i) {
            int idx = std::min(static_cast<int>(std::round(i * step)), static_cast<int>(deduped.size()) - 1);
            compact.push_back(deduped[idx]);
        }
        deduped = compact;
    }

    for (int i = 0; i < 6; ++i) {
        selected[i] = deduped[i];
    }
    return selected;
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

    try {
        result.mask = buildColorMask(input, color);

        fillMaskHoles(result.mask);
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::morphologyEx(result.mask, result.mask, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(result.mask, result.mask, cv::MORPH_CLOSE, kernel);
        fillMaskHoles(result.mask);

        HexagonCandidate hexagon = extractLargestHexagon(result.mask);
        if (!hexagon.valid) {
            result.annotated = input.clone();
            return result;
        }

        PoseCandidate pose = solvePoseFromHexagon(hexagon.corners);
        if (!pose.valid) {
            result.annotated = drawAnnotation(input, hexagon.contour, hexagon.corners, {}, cv::Vec3d(), cv::Vec3d(), -1.0);
            result.hexagonCorners = hexagon.corners;
            result.imageCorners2d.assign(hexagon.corners.begin(), hexagon.corners.end());
            return result;
        }

        cv::Mat rotationMatrix;
        cv::Rodrigues(pose.rvec, rotationMatrix);
        result.rotationMatrix = rotationMatrix;
        result.found = true;
        result.hexagonCorners = pose.imageCorners;
        result.imageCorners2d.assign(pose.imageCorners.begin(), pose.imageCorners.end());
        result.objectCorners3d = pose.objectCorners;
        result.cameraCorners3d = transformToCameraCoordinates(pose.objectCorners, result.rotationMatrix, pose.tvec);
        result.projectedCubeCorners2d = pose.projectedCorners;
        result.rvec = pose.rvec;
        result.tvec = pose.tvec;
        result.reprojectionError = pose.reprojectionError;
        result.annotated = drawAnnotation(input,
                                          hexagon.contour,
                                          pose.imageCorners,
                                          pose.projectedCorners,
                                          pose.rvec,
                                          pose.tvec,
                                          pose.reprojectionError);
        return result;
    } catch (const cv::Exception&) {
        result.annotated = input.clone();
        return result;
    }
}

cv::Mat CubeTracker3D::applyClaheToValueChannel(const cv::Mat& input) {
    cv::Mat lab;
    cv::cvtColor(input, lab, cv::COLOR_BGR2Lab);

    std::vector<cv::Mat> channels;
    cv::split(lab, channels);

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.5, cv::Size(8, 8));
    clahe->apply(channels[0], channels[0]);

    cv::merge(channels, lab);
    cv::Mat enhanced;
    cv::cvtColor(lab, enhanced, cv::COLOR_Lab2BGR);
    return enhanced;
}

cv::Mat CubeTracker3D::buildColorMask(const cv::Mat& input, const std::string& color) const {
    cv::Mat enhanced = applyClaheToValueChannel(input);
    cv::Mat hsv;
    cv::cvtColor(enhanced, hsv, cv::COLOR_BGR2HSV);

    std::vector<cv::Mat> hsvChannels;
    cv::split(hsv, hsvChannels);
    const cv::Mat& h = hsvChannels[0];
    const cv::Mat& s = hsvChannels[1];
    const cv::Mat& v = hsvChannels[2];

    std::vector<cv::Mat> bgrChannels;
    cv::split(enhanced, bgrChannels);
    const cv::Mat& b = bgrChannels[0];
    const cv::Mat& g = bgrChannels[1];
    const cv::Mat& r = bgrChannels[2];

    cv::Mat mask1, mask2, baseMask, hueMask, satMask, valMask;
    cv::Mat dominantMask, washedOutMask, mask;

    if (color == "red") {
        cv::inRange(hsv, cv::Scalar(0, 45, 40), cv::Scalar(12, 255, 255), mask1);
        cv::inRange(hsv, cv::Scalar(165, 45, 40), cv::Scalar(180, 255, 255), mask2);
        cv::bitwise_or(mask1, mask2, baseMask);

        cv::inRange(h, 0, 18, mask1);
        cv::inRange(h, 160, 180, mask2);
        cv::bitwise_or(mask1, mask2, hueMask);
        cv::inRange(s, 10, 150, satMask);
        cv::inRange(v, 170, 255, valMask);
        cv::bitwise_and(hueMask, satMask, washedOutMask);
        cv::bitwise_and(washedOutMask, valMask, washedOutMask);

        cv::Mat gbMax;
        cv::max(g, b, gbMax);
        cv::Mat gbMaxOffset;
        cv::add(gbMax, cv::Scalar(18), gbMaxOffset);
        cv::compare(r, gbMaxOffset, dominantMask, cv::CMP_GT);
    } else {
        cv::inRange(hsv, cv::Scalar(88, 40, 40), cv::Scalar(138, 255, 255), baseMask);

        cv::inRange(h, 84, 145, hueMask);
        cv::inRange(s, 8, 145, satMask);
        cv::inRange(v, 160, 255, valMask);
        cv::bitwise_and(hueMask, satMask, washedOutMask);
        cv::bitwise_and(washedOutMask, valMask, washedOutMask);

        cv::Mat rgMax;
        cv::max(r, g, rgMax);
        cv::Mat rgMaxOffset;
        cv::add(rgMax, cv::Scalar(15), rgMaxOffset);
        cv::compare(b, rgMaxOffset, dominantMask, cv::CMP_GT);
    }

    cv::bitwise_or(baseMask, washedOutMask, mask);
    cv::bitwise_or(mask, dominantMask, mask);
    return mask;
}

void CubeTracker3D::fillMaskHoles(cv::Mat& mask) {
    if (mask.empty()) {
        return;
    }

    cv::Mat flood = mask.clone();
    cv::floodFill(flood, cv::Point(0, 0), cv::Scalar(255));
    cv::Mat floodInv;
    cv::bitwise_not(flood, floodInv);
    mask |= floodInv;
}

CubeTracker3D::HexagonCandidate CubeTracker3D::extractLargestHexagon(const cv::Mat& mask) const {
    HexagonCandidate best;

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    double largestArea = 0.0;
    std::vector<cv::Point> largestContour;
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area > largestArea) {
            largestArea = area;
            largestContour = contour;
        }
    }

    if (largestContour.empty() || largestArea < kMinContourArea) {
        return best;
    }

    double perimeter = cv::arcLength(largestContour, true);
    std::vector<cv::Point> approx;

    for (double alpha = 0.005; alpha <= 0.08; alpha += 0.0025) {
        cv::approxPolyDP(largestContour, approx, alpha * perimeter, true);
        if (approx.size() == kPolygonVertexCount) {
            std::vector<cv::Point2f> points;
            points.reserve(approx.size());
            for (const auto& point : approx) {
                points.emplace_back(static_cast<float>(point.x), static_cast<float>(point.y));
            }

            best.valid = true;
            best.area = largestArea;
            best.contour = largestContour;
            best.corners = orderPolygonCorners(points);
            return best;
        }
        if (approx.size() > kPolygonVertexCount) {
            std::vector<cv::Point2f> reduced = reducePolygonToSix(approx);
            if (reduced.size() == kPolygonVertexCount) {
                best.valid = true;
                best.area = largestArea;
                best.contour = largestContour;
                best.corners = orderPolygonCorners(reduced);
                return best;
            }
        }
    }

    std::vector<cv::Point> hull;
    cv::convexHull(largestContour, hull);
    if (hull.size() >= kPolygonVertexCount) {
        for (double alpha = 0.005; alpha <= 0.08; alpha += 0.0025) {
            cv::approxPolyDP(hull, approx, alpha * cv::arcLength(hull, true), true);
            if (approx.size() == kPolygonVertexCount) {
                std::vector<cv::Point2f> points;
                points.reserve(approx.size());
                for (const auto& point : approx) {
                    points.emplace_back(static_cast<float>(point.x), static_cast<float>(point.y));
                }

                best.valid = true;
                best.area = largestArea;
                best.contour = largestContour;
                best.corners = orderPolygonCorners(points);
                return best;
            }
            if (approx.size() > kPolygonVertexCount) {
                std::vector<cv::Point2f> reduced = reducePolygonToSix(approx);
                if (reduced.size() == kPolygonVertexCount) {
                    best.valid = true;
                    best.area = largestArea;
                    best.contour = largestContour;
                    best.corners = orderPolygonCorners(reduced);
                    return best;
                }
            }
        }
    }

    std::array<cv::Point2f, 6> fallbackCorners = selectSixCornersFromContour(largestContour);
    bool fallbackValid = true;
    for (const auto& p : fallbackCorners) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || (std::abs(p.x) < 1e-3f && std::abs(p.y) < 1e-3f)) {
            fallbackValid = false;
            break;
        }
    }
    if (fallbackValid) {
        std::vector<cv::Point2f> points(fallbackCorners.begin(), fallbackCorners.end());
        best.valid = true;
        best.area = largestArea;
        best.contour = largestContour;
        best.corners = orderPolygonCorners(points);
        return best;
    }

    return best;
}

std::array<cv::Point2f, 6> CubeTracker3D::orderPolygonCorners(const std::vector<cv::Point2f>& points) {
    std::array<cv::Point2f, 6> ordered{};
    if (points.size() != kPolygonVertexCount) {
        return ordered;
    }

    cv::Point2f center = averagePoint(points);
    std::vector<cv::Point2f> sorted = points;
    std::sort(sorted.begin(),
              sorted.end(),
              [center](const cv::Point2f& lhs, const cv::Point2f& rhs) {
                  return std::atan2(lhs.y - center.y, lhs.x - center.x) < std::atan2(rhs.y - center.y, rhs.x - center.x);
              });

    int startIndex = 0;
    for (int i = 1; i < static_cast<int>(sorted.size()); ++i) {
        if (sorted[i].y < sorted[startIndex].y ||
            (std::abs(sorted[i].y - sorted[startIndex].y) < 1e-3f && sorted[i].x < sorted[startIndex].x)) {
            startIndex = i;
        }
    }

    for (int i = 0; i < kPolygonVertexCount; ++i) {
        ordered[i] = sorted[(startIndex + i) % kPolygonVertexCount];
    }
    return ordered;
}

std::vector<CubeTracker3D::EdgeInfo> CubeTracker3D::buildEdgeInfos(const std::array<cv::Point2f, 6>& corners) {
    std::vector<EdgeInfo> edges;
    edges.reserve(kPolygonVertexCount);

    for (int i = 0; i < kPolygonVertexCount; ++i) {
        const cv::Point2f& a = corners[i];
        const cv::Point2f& b = corners[(i + 1) % kPolygonVertexCount];
        cv::Point2f delta = b - a;
        float length = std::sqrt(delta.dot(delta));
        if (length < 1e-3f) {
            continue;
        }

        EdgeInfo edge;
        edge.startIndex = i;
        edge.endIndex = (i + 1) % kPolygonVertexCount;
        edge.length = length;
        edge.direction = delta * (1.0f / length);
        edge.angle = wrapAngle(std::atan2(delta.y, delta.x));
        edge.line = cv::Vec3f(a.y - b.y, b.x - a.x, a.x * b.y - b.x * a.y);
        edges.push_back(edge);
    }

    return edges;
}

bool CubeTracker3D::clusterEdges(std::vector<EdgeInfo>& edges) {
    if (edges.size() != kPolygonVertexCount) {
        return false;
    }

    std::array<int, 6> labels{};
    labels.fill(-1);
    int clusterId = 0;

    for (int i = 0; i < kPolygonVertexCount; ++i) {
        if (labels[i] != -1) {
            continue;
        }
        labels[i] = clusterId;
        for (int j = i + 1; j < kPolygonVertexCount; ++j) {
            float diff = std::abs(edges[i].angle - edges[j].angle);
            diff = std::min(diff, static_cast<float>(CV_PI) - diff);
            if (diff < 0.30f) {
                labels[j] = clusterId;
            }
        }
        ++clusterId;
    }

    if (clusterId != 3) {
        for (int i = 0; i < kPolygonVertexCount; ++i) {
            edges[i].cluster = i % 3;
        }
        return true;
    }

    std::array<int, 3> counts{};
    counts.fill(0);
    for (int i = 0; i < kPolygonVertexCount; ++i) {
        if (labels[i] < 0 || labels[i] >= 3) {
            return false;
        }
        counts[labels[i]]++;
    }

    if (counts[0] != 2 || counts[1] != 2 || counts[2] != 2) {
        for (int i = 0; i < kPolygonVertexCount; ++i) {
            edges[i].cluster = i % 3;
        }
        return true;
    }

    for (int i = 0; i < kPolygonVertexCount; ++i) {
        edges[i].cluster = labels[i];
    }

    return true;
}

std::vector<cv::Point2f> CubeTracker3D::estimateVanishingPoints(const std::vector<EdgeInfo>& edges) {
    std::vector<cv::Point2f> vanishingPoints(3, cv::Point2f(std::numeric_limits<float>::quiet_NaN(),
                                                            std::numeric_limits<float>::quiet_NaN()));

    for (int cluster = 0; cluster < 3; ++cluster) {
        std::vector<const EdgeInfo*> clusterEdges;
        for (const auto& edge : edges) {
            if (edge.cluster == cluster) {
                clusterEdges.push_back(&edge);
            }
        }

        if (clusterEdges.size() != 2) {
            continue;
        }

        cv::Point2f intersection;
        if (intersectLines(clusterEdges[0]->line, clusterEdges[1]->line, &intersection)) {
            vanishingPoints[cluster] = intersection;
        }
    }

    return vanishingPoints;
}

std::array<cv::Point2f, 6> CubeTracker3D::rotateCorners(const std::array<cv::Point2f, 6>& corners, int offset, bool reverse) {
    std::array<cv::Point2f, 6> rotated{};
    for (int i = 0; i < kPolygonVertexCount; ++i) {
        int index = reverse ? (offset - i + kPolygonVertexCount * 2) % kPolygonVertexCount : (offset + i) % kPolygonVertexCount;
        rotated[i] = corners[index];
    }
    return rotated;
}

std::vector<cv::Point3f> CubeTracker3D::buildCubeObjectPoints() const {
    const float halfWidth = originWidth_ * 0.5f;
    const float halfHeight = originHeight_ * 0.5f;
    const float halfDepth = originDepth_ * 0.5f;

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

CubeTracker3D::PoseCandidate CubeTracker3D::solvePoseFromHexagon(const std::array<cv::Point2f, 6>& corners) const {
    PoseCandidate best;

    std::vector<EdgeInfo> edges = buildEdgeInfos(corners);
    if (!clusterEdges(edges)) {
        return best;
    }

    const std::vector<cv::Point2f> vanishingPoints = estimateVanishingPoints(edges);
    const cv::Point2f center = averagePoint(std::vector<cv::Point2f>(corners.begin(), corners.end()));
    const double contourArea = polygonArea(corners);

    const std::vector<cv::Point3f> cubePoints = buildCubeObjectPoints();
    const std::array<std::array<int, 6>, 8> visibleCycles = {{
        {0, 1, 2, 6, 7, 4},
        {1, 2, 3, 7, 4, 5},
        {2, 3, 0, 4, 5, 6},
        {3, 0, 1, 5, 6, 7},
        {4, 5, 6, 2, 1, 0},
        {5, 6, 7, 3, 2, 1},
        {6, 7, 4, 0, 3, 2},
        {7, 4, 5, 1, 0, 3},
    }};

    for (const auto& visibleIndexCycle : visibleCycles) {
        for (int offset = 0; offset < kPolygonVertexCount; ++offset) {
            for (bool reverse : {false, true}) {
                std::array<cv::Point2f, 6> orderedCorners = rotateCorners(corners, offset, reverse);

                std::vector<cv::Point3f> objectPoints;
                objectPoints.reserve(kPolygonVertexCount);
                for (int index : visibleIndexCycle) {
                    objectPoints.push_back(cubePoints[index]);
                }

                std::vector<cv::Point2f> imagePoints(orderedCorners.begin(), orderedCorners.end());

                cv::Vec3d rvec;
                cv::Vec3d tvec;
                bool ok = cv::solvePnPRansac(objectPoints,
                                             imagePoints,
                                             cameraMatrix_,
                                             distCoeffs_,
                                             rvec,
                                             tvec,
                                             false,
                                             200,
                                             6.0,
                                             0.99);
                if (!ok) {
                    ok = cv::solvePnP(objectPoints, imagePoints, cameraMatrix_, distCoeffs_, rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);
                }
                if (!ok) {
                    continue;
                }

                double reprojectionError = computeReprojectionError(objectPoints, imagePoints, rvec, tvec);
                if (!std::isfinite(reprojectionError)) {
                    continue;
                }

                std::vector<cv::Point2f> projectedCorners;
                cv::projectPoints(cubePoints, rvec, tvec, cameraMatrix_, distCoeffs_, projectedCorners);
                if (projectedCorners.size() != cubePoints.size()) {
                    continue;
                }

                std::array<cv::Point2f, 6> projectedVisible{};
                for (int i = 0; i < kPolygonVertexCount; ++i) {
                    projectedVisible[i] = projectedCorners[visibleIndexCycle[i]];
                }

                std::vector<EdgeInfo> projectedEdges = buildEdgeInfos(projectedVisible);
                if (!clusterEdges(projectedEdges)) {
                    continue;
                }
                const std::vector<cv::Point2f> projectedVanishing = estimateVanishingPoints(projectedEdges);

                double vpScore = 0.0;
                for (size_t i = 0; i < vanishingPoints.size(); ++i) {
                    vpScore += distanceOrLarge(vanishingPoints[i], projectedVanishing[i]);
                }

                const cv::Point2f projectedCenter =
                    averagePoint(std::vector<cv::Point2f>(projectedVisible.begin(), projectedVisible.end()));
                const double centerScore = cv::norm(center - projectedCenter);
                const double areaScore = std::abs(polygonArea(projectedVisible) - contourArea) / (contourArea + 1e-6);
                const double totalScore = reprojectionError + 0.001 * vpScore + 0.02 * centerScore + 20.0 * areaScore;

                if (totalScore < best.totalScore) {
                    best.valid = true;
                    best.imageCorners = orderedCorners;
                    best.objectCorners = std::move(objectPoints);
                    best.projectedCorners = std::move(projectedCorners);
                    best.rvec = rvec;
                    best.tvec = tvec;
                    best.reprojectionError = reprojectionError;
                    best.totalScore = totalScore;
                }
            }
        }
    }

    return best;
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

double CubeTracker3D::computeReprojectionError(const std::vector<cv::Point3f>& objectPoints,
                                               const std::vector<cv::Point2f>& imagePoints,
                                               const cv::Vec3d& rvec,
                                               const cv::Vec3d& tvec) const {
    if (objectPoints.empty() || objectPoints.size() != imagePoints.size()) {
        return std::numeric_limits<double>::infinity();
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
                                      const std::vector<cv::Point>& contour,
                                      const std::array<cv::Point2f, 6>& hexagonCorners,
                                      const std::vector<cv::Point2f>& projectedCorners,
                                      const cv::Vec3d& rvec,
                                      const cv::Vec3d& tvec,
                                      double reprojectionError) const {
    cv::Mat annotated = input.clone();

    if (!contour.empty()) {
        cv::drawContours(annotated, std::vector<std::vector<cv::Point>>{contour}, -1, cv::Scalar(0, 180, 255), 2);
    }

    std::vector<cv::Point> hexagon;
    hexagon.reserve(kPolygonVertexCount);
    for (int i = 0; i < kPolygonVertexCount; ++i) {
        hexagon.emplace_back(cvRound(hexagonCorners[i].x), cvRound(hexagonCorners[i].y));
        cv::circle(annotated, hexagonCorners[i], 5, cv::Scalar(0, 255, 255), -1);
        cv::putText(annotated,
                    std::to_string(i),
                    hexagonCorners[i] + cv::Point2f(4.0f, -4.0f),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    cv::Scalar(0, 255, 255),
                    1);
    }
    if (!hexagon.empty()) {
        cv::polylines(annotated, hexagon, true, cv::Scalar(0, 255, 0), 2);
    }

    if (!projectedCorners.empty()) {
        for (size_t i = 0; i < projectedCorners.size(); ++i) {
            cv::circle(annotated, projectedCorners[i], 3, cv::Scalar(255, 0, 255), -1);
        }
    }

    if (reprojectionError >= 0.0) {
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
    }

    return annotated;
}

std::string CubeTracker3D::vecToString(const cv::Vec3d& v) {
    std::ostringstream oss;
    oss << v[0] << ", " << v[1] << ", " << v[2];
    return oss.str();
}

}  // namespace tracker
