// KFS-Tracker ROS 2 节点：复用 kfs_tracker_core 的 CameraProvider 与 tracker 推理器，
// 支持「本地直连摄像头」与「订阅 ROS 图像话题」两种输入，发布 KFSDetection 结果与标注图。
//
// 单线程 executor 串行调度：timer / topic 回调 / 推理都在同一线程，
// 满足推理器（PIMPL，非线程安全）的不跨线程约束，无需任何锁。

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include <chrono>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

#include <opencv2/opencv.hpp>

#include "bbox_tracking_classify.h"
#include "camera_provider.h"
#include "heatmap_classifier_infer.h"
#include "heatmap_tracking_classify.h"
#include "infer_pipeline.h"
#include "kfs_tracker/msg/kfs_detection.hpp"

class KFSTrackerNode : public rclcpp::Node {
public:
  KFSTrackerNode() : rclcpp::Node("kfs_tracker_node") {
    const std::string share = ament_index_cpp::get_package_share_directory("kfs_tracker");

    // ---- 通用参数（构造期读出存成员）----
    input_source_ = declare_parameter<std::string>("input_source", "topic");  // "local" | "topic"
    model_type_   = declare_parameter<std::string>("model_type", "heatmap");  // "heatmap" | "bbox"
    input_size_   = declare_parameter<int>("input_size", 300);
    display_ui_   = declare_parameter<bool>("display_ui", false);
    publish_annotated_ = declare_parameter<bool>("publish_annotated", true);
    frame_id_     = declare_parameter<std::string>("frame_id", "kfs_camera");
    // 默认模型路径指向 install 后的 share/model 目录，可被参数覆盖
    const std::string default_model = share + (model_type_ == "heatmap"
        ? "/model/corner_heatmap_localizer.pt"
        : "/model/tracker_localizer.pt");
    model_path_   = declare_parameter<std::string>("model", default_model);
    classifier_model_path_ =
        declare_parameter<std::string>("classifier_model", share + "/model/tracker_classifier.pt");

    // ---- 其余参数：统一声明（供 yaml 注入不报 undclared），值在 init* 中读取 ----
    declare_parameter<std::string>("image_topic", "/camera/color/image_raw");
    declare_parameter<std::string>("camera_info_topic", "/camera/color/camera_info");
    declare_parameter<std::string>("depth_topic", "/camera/aligned_depth_to_color/image_raw");
    declare_parameter<bool>("use_compressed", false);
    declare_parameter<bool>("use_depth", false);
    declare_parameter<std::string>("camera_type", "opencv");   // opencv | realsense | file
    declare_parameter<int>("camera_index", 0);
    declare_parameter<std::string>("video_path", "");
    declare_parameter<std::string>("intrinsics_path", "");
    declare_parameter<int>("frame_width", 1280);
    declare_parameter<int>("frame_height", 720);
    declare_parameter<double>("grab_fps", 30.0);
    declare_parameter<std::string>("detection_topic", "/kfs_tracker/detection");
    declare_parameter<std::string>("annotated_topic", "/kfs_tracker/annotated_image");

    // ---- 推理器 ----
    initPredictor();

    // ---- 发布器 ----
    const std::string det_topic = get_parameter("detection_topic").as_string();
    const std::string ann_topic = get_parameter("annotated_topic").as_string();
    det_pub_ = create_publisher<kfs_tracker::msg::KFSDetection>(det_topic, 10);
    if (publish_annotated_) {
      annotated_pub_ = create_publisher<sensor_msgs::msg::Image>(ann_topic, rclcpp::SensorDataQoS());
    }

    // ---- 输入源接线 ----
    if (input_source_ == "local") {
      initLocalProvider();
    } else {
      initTopicSubscribers();
    }

    RCLCPP_INFO(get_logger(),
                "ready: input_source=%s model_type=%s%s",
                input_source_.c_str(), model_type_.c_str(),
                display_ui_ ? " (display_ui)" : "");
  }

private:
  // ==================== 推理器构造 ====================
  void initPredictor() {
    classifierInfer_ =
        std::make_unique<tracker::ImageClassifierInfer>(classifier_model_path_, input_size_);
    if (model_type_ == "heatmap") {
      predictor_ = std::make_unique<kfs::Predictor>(
          std::in_place_type<tracker::HeatmapTrackingClassifier>, model_path_, input_size_);
    } else {
      predictor_ = std::make_unique<kfs::Predictor>(
          std::in_place_type<tracker::BboxTrackingClassifier>, model_path_, input_size_);
    }
  }

  // ==================== 统一帧处理：两条输入路径都汇到这里 ====================
  void processFrame(const cv::Mat& color, const cv::Mat& depth,
                    const camera::Intrinsics& /*intr*/,
                    const rclcpp::Time& stamp, const std::string& frame_id) {
    if (color.empty()) return;

    // 1. 定位 / 角点推理（单次，携带 corners/bbox，避免二次推理）
    auto out = kfs::inferAnnotated(*predictor_, color);
    // 2. 分类 + 绘制
    const auto cls = classifierInfer_->infer(color);
    kfs::drawClassification(out.annotated, kfs::formatClassification(cls));

    // 3. 组装 KFSDetection
    kfs_tracker::msg::KFSDetection msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = frame_id;
    msg.valid = out.valid;
    msg.model_type = model_type_;
    msg.class_valid = cls.valid;
    msg.class_id = cls.classId;
    msg.class_name = cls.className;
    msg.class_score = cls.classScore;
    msg.infer_ms = static_cast<float>(out.inferMs);

    float cx = 0.f, cy = 0.f;
    if (out.isHeatmap && out.valid) {
      float xmin = 1e9f, ymin = 1e9f, xmax = -1e9f, ymax = -1e9f;
      for (size_t i = 0; i < msg.corners.size(); ++i) {
        msg.corners[i].x = out.corners[i].x;
        msg.corners[i].y = out.corners[i].y;
        xmin = std::min(xmin, out.corners[i].x);
        ymin = std::min(ymin, out.corners[i].y);
        xmax = std::max(xmax, out.corners[i].x);
        ymax = std::max(ymax, out.corners[i].y);
      }
      msg.bbox_min.x = xmin; msg.bbox_min.y = ymin;
      msg.bbox_max.x = xmax; msg.bbox_max.y = ymax;
      cx = (xmin + xmax) * 0.5f; cy = (ymin + ymax) * 0.5f;
    } else if (!out.isHeatmap && out.valid) {
      msg.bbox_min.x = static_cast<float>(out.bbox.x);
      msg.bbox_min.y = static_cast<float>(out.bbox.y);
      msg.bbox_max.x = static_cast<float>(out.bbox.x + out.bbox.width);
      msg.bbox_max.y = static_cast<float>(out.bbox.y + out.bbox.height);
      cx = out.bbox.x + out.bbox.width * 0.5f;
      cy = out.bbox.y + out.bbox.height * 0.5f;
    }

    // 4. 中心距离（仅当输入带 16UC1 mm 深度图）
    msg.center_distance_m = -1.0f;
    if (out.valid && !depth.empty() && depth.type() == CV_16U) {
      const int ix = static_cast<int>(cx), iy = static_cast<int>(cy);
      if (ix >= 0 && iy >= 0 && ix < depth.cols && iy < depth.rows) {
        const uint16_t mm = depth.at<uint16_t>(iy, ix);
        if (mm != 0) msg.center_distance_m = mm / 1000.0f;
      }
    }

    det_pub_->publish(msg);

    // 5. 可选标注图发布（OpenCV UI → topic）
    if (publish_annotated_ && !out.annotated.empty()) {
      auto img_msg = cv_bridge::CvImage(msg.header, sensor_msgs::image_encodings::BGR8,
                                        out.annotated).toImageMsg();
      annotated_pub_->publish(*img_msg);
    }

    // 6. 可选本地显示（display_ui）
    if (display_ui_ && !out.annotated.empty()) {
      cv::imshow("kfs_tracker_node", out.annotated);
      cv::waitKey(1);
    }
  }

  // ==================== topic 输入模式 ====================
  void initTopicSubscribers() {
    const std::string image_topic = get_parameter("image_topic").as_string();
    const std::string info_topic  = get_parameter("camera_info_topic").as_string();
    const std::string depth_topic = get_parameter("depth_topic").as_string();
    const bool use_compressed = get_parameter("use_compressed").as_bool();
    const bool use_depth      = get_parameter("use_depth").as_bool();

    // CameraInfo：总是订阅（可选；无发布者则 latest_intrinsics_ 保持兜底）
    info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
        info_topic, rclcpp::SensorDataQoS(),
        std::bind(&KFSTrackerNode::onCameraInfo, this, std::placeholders::_1));

    if (use_depth) {
      // image + depth 近似时间同步（均为 raw Image）
      mf_image_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(
          this, image_topic, rmw_qos_profile_sensor_data);
      mf_depth_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(
          this, depth_topic, rmw_qos_profile_sensor_data);
      using Policy = message_filters::sync_policies::ApproximateTime<
          sensor_msgs::msg::Image, sensor_msgs::msg::Image>;
      sync_depth_ = std::make_shared<message_filters::Synchronizer<Policy>>(
          Policy(10), *mf_image_, *mf_depth_);
      sync_depth_->registerCallback(&KFSTrackerNode::onSyncedImageDepth, this);
      RCLCPP_INFO(get_logger(), "topic input: %s + depth %s (sync)", image_topic.c_str(), depth_topic.c_str());
    } else if (use_compressed) {
      comp_sub_ = create_subscription<sensor_msgs::msg::CompressedImage>(
          image_topic, rclcpp::SensorDataQoS(),
          std::bind(&KFSTrackerNode::onImageCompressed, this, std::placeholders::_1));
      RCLCPP_INFO(get_logger(), "topic input (compressed): %s", image_topic.c_str());
    } else {
      raw_sub_ = create_subscription<sensor_msgs::msg::Image>(
          image_topic, rclcpp::SensorDataQoS(),
          std::bind(&KFSTrackerNode::onImageRaw, this, std::placeholders::_1));
      RCLCPP_INFO(get_logger(), "topic input (raw): %s", image_topic.c_str());
    }
  }

  void onCameraInfo(const sensor_msgs::msg::CameraInfo::ConstSharedPtr& msg) {
    camera::Intrinsics intr;
    cv::Mat K(3, 3, CV_64F);
    for (int i = 0; i < 9; ++i) K.at<double>(i) = msg->k[i];
    cv::Mat D(1, static_cast<int>(msg->d.size()), CV_64F);
    for (size_t i = 0; i < msg->d.size(); ++i) D.at<double>(static_cast<int>(i)) = msg->d[i];
    intr.cameraMatrix = K;
    intr.distCoeffs = D;
    latest_intrinsics_ = intr;
  }

  void onImageRaw(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    } catch (const cv_bridge::Exception& e) {
      RCLCPP_ERROR(get_logger(), "cv_bridge(raw): %s", e.what());
      return;
    }
    processFrame(cv_ptr->image, cv::Mat(), latest_intrinsics_, msg->header.stamp,
                 msg->header.frame_id.empty() ? frame_id_ : msg->header.frame_id);
  }

  void onImageCompressed(const sensor_msgs::msg::CompressedImage::ConstSharedPtr& msg) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    } catch (const cv_bridge::Exception& e) {
      RCLCPP_ERROR(get_logger(), "cv_bridge(compressed): %s", e.what());
      return;
    }
    processFrame(cv_ptr->image, cv::Mat(), latest_intrinsics_, msg->header.stamp, frame_id_);
  }

  void onSyncedImageDepth(const sensor_msgs::msg::Image::ConstSharedPtr& img,
                          const sensor_msgs::msg::Image::ConstSharedPtr& dep) {
    cv::Mat color, depth;
    try {
      color = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::BGR8)->image;
    } catch (const cv_bridge::Exception& e) {
      RCLCPP_ERROR(get_logger(), "cv_bridge(color): %s", e.what());
      return;
    }
    try {
      depth = cv_bridge::toCvCopy(dep, dep->encoding)->image;  // 16UC1 mm 保持原编码
    } catch (const cv_bridge::Exception& e) {
      RCLCPP_ERROR(get_logger(), "cv_bridge(depth): %s", e.what());
      depth = cv::Mat();
    }
    processFrame(color, depth, latest_intrinsics_, img->header.stamp,
                 img->header.frame_id.empty() ? frame_id_ : img->header.frame_id);
  }

  // ==================== local 输入模式 ====================
  void initLocalProvider() {
    camera::CameraConfig cfg;
    cfg.type           = get_parameter("camera_type").as_string();
    cfg.cameraIndex    = get_parameter("camera_index").as_int();
    cfg.videoPath      = get_parameter("video_path").as_string();
    cfg.intrinsicsPath = get_parameter("intrinsics_path").as_string();
    cfg.width          = get_parameter("frame_width").as_int();
    cfg.height         = get_parameter("frame_height").as_int();
    const double grab_fps = get_parameter("grab_fps").as_double();

    provider_ = camera::createCameraProvider(cfg);
    if (!provider_) {
      RCLCPP_FATAL(get_logger(), "createCameraProvider failed for type '%s'", cfg.type.c_str());
      throw std::runtime_error("camera provider init failed");
    }

    // 内参：有 YAML 用真实标定，否则兜底 identity
    if (cfg.intrinsicsPath.empty()) {
      latest_intrinsics_ = camera::makeDefaultIntrinsics(cfg.width, cfg.height);
    } else {
      try {
        latest_intrinsics_ = camera::loadIntrinsicsFromFile(cfg.intrinsicsPath);
      } catch (const std::exception& e) {
        RCLCPP_WARN(get_logger(), "load intrinsics failed: %s; using default", e.what());
        latest_intrinsics_ = camera::makeDefaultIntrinsics(cfg.width, cfg.height);
      }
    }

    // wall_timer 驱动 grab 循环；单线程 executor 下与推理串行，无锁
    const auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(1.0 / grab_fps));
    grab_timer_ = create_wall_timer(period, [this]() { onGrabTimer(); });
    RCLCPP_INFO(get_logger(), "local input: %s @ %dx%d (%.1f fps)",
                cfg.type.c_str(), cfg.width, cfg.height, grab_fps);
  }

  void onGrabTimer() {
    camera::FrameBundle bundle;
    if (!provider_->grab(bundle)) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "grab returned false");
      return;
    }
    processFrame(bundle.color, bundle.depth, bundle.intrinsics, now(), frame_id_);
  }

private:
  // 推理
  std::unique_ptr<kfs::Predictor> predictor_;
  std::unique_ptr<tracker::ImageClassifierInfer> classifierInfer_;
  camera::Intrinsics latest_intrinsics_{};

  // 通用配置
  std::string input_source_;
  std::string model_type_;
  std::string model_path_;
  std::string classifier_model_path_;
  std::string frame_id_;
  int input_size_ = 300;
  bool display_ui_ = false;
  bool publish_annotated_ = true;

  // 发布
  rclcpp::Publisher<kfs_tracker::msg::KFSDetection>::SharedPtr det_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr annotated_pub_;

  // local 模式
  std::unique_ptr<camera::CameraProvider> provider_;
  rclcpp::TimerBase::SharedPtr grab_timer_;

  // topic 模式
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr raw_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr comp_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;
  std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> mf_image_;
  std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> mf_depth_;
  std::shared_ptr<message_filters::Synchronizer<
      message_filters::sync_policies::ApproximateTime<
          sensor_msgs::msg::Image, sensor_msgs::msg::Image>>> sync_depth_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<KFSTrackerNode>());
  } catch (const std::exception& e) {
    std::cerr << "[kfs_tracker_node] fatal: " << e.what() << std::endl;
  }
  rclcpp::shutdown();
  return 0;
}
