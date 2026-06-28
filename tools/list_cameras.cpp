// list_cameras: 独立小工具，枚举所有摄像头（OpenCV + RealSense），
// 从每个设备抓一帧，标注 index 和设备信息，拼成一张图保存。
//
// 编译（不依赖 LibTorch，只需 OpenCV + librealsense2）：
//   g++ -std=c++17 -O2 tools/list_cameras.cpp -o tools/list_cameras \
//       $(pkg-config --cflags --libs opencv4) -lrealsense2
//
// 运行：
//   ./tools/list_cameras [--output camera_overview.jpg]

#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>
#include <vector>

#ifdef ENABLE_REALSENSE
#include <librealsense2/rs.hpp>
#endif

// ─── 辅助：命令行参数 ───
static std::string getArg(int argc, char** argv, const std::string& key, const std::string& def) {
    for (int i = 1; i + 1 < argc; ++i)
        if (key == argv[i]) return argv[i + 1];
    return def;
}

// ─── 在帧左上角画标签 ───
static void drawLabel(cv::Mat& frame, const std::string& text) {
    if (frame.type() != CV_8UC3 || frame.empty()) return;
    int baseline = 0;
    double scale = 0.7;
    int thick = 2;
    cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, scale, thick, &baseline);
    cv::rectangle(frame, {0, 0}, {ts.width + 15, ts.height + 15}, {0, 0, 0}, -1);
    cv::putText(frame, text, {7, ts.height + 8}, cv::FONT_HERSHEY_SIMPLEX, scale, {0, 255, 255}, thick);
}

// ─── 强制转为 CV_8UC3 ───
static cv::Mat ensureBGR(const cv::Mat& src) {
    if (src.type() == CV_8UC3) return src.clone();
    cv::Mat dst;
    if (src.channels() == 1)
        cv::cvtColor(src, dst, cv::COLOR_GRAY2BGR);
    else if (src.channels() == 4)
        cv::cvtColor(src, dst, cv::COLOR_BGRA2BGR);
    else {
        src.convertTo(dst, CV_8U);
        if (dst.channels() != 3) cv::cvtColor(dst, dst, cv::COLOR_GRAY2BGR);
    }
    return dst;
}

int main(int argc, char** argv) {
    std::string outputPath = getArg(argc, argv, "--output", "camera_overview.jpg");

    struct Snap { std::string label; cv::Mat frame; };
    std::vector<Snap> snaps;

    // ─── 1. OpenCV 摄像头（/dev/video0~9）───
    std::cout << "=== Scanning OpenCV cameras ===" << std::endl;
    for (int i = 0; i < 10; ++i) {
        cv::VideoCapture cap(i, cv::CAP_V4L2);
        if (!cap.isOpened()) continue;
        cap.set(cv::CAP_PROP_CONVERT_RGB, 1);
        cv::Mat frame;
        // 丢弃前几帧让自动曝光稳定
        for (int w = 0; w < 10; ++w) { cap >> frame; if (!frame.empty()) break; }
        cap >> frame;
        cap.release();
        if (frame.empty()) continue;

        frame = ensureBGR(frame);
        std::string label = "opencv:" + std::to_string(i);
        std::cout << "  [" << label << "] " << frame.cols << "x" << frame.rows << std::endl;
        drawLabel(frame, label);
        snaps.push_back({label, frame});
    }

    // ─── 2. RealSense 设备 ───
#ifdef ENABLE_REALSENSE
    std::cout << "=== Scanning RealSense cameras ===" << std::endl;
    {
        rs2::context ctx;
        auto devs = ctx.query_devices();
        for (size_t i = 0; i < devs.size(); ++i) {
            std::string serial = devs[i].get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
            std::string name   = devs[i].get_info(RS2_CAMERA_INFO_NAME);
            try {
                rs2::pipeline pipe;
                rs2::config cfg;
                cfg.enable_device(serial);
                cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
                pipe.start(cfg);
                rs2::frameset fs;
                for (int t = 0; t < 10; ++t) { fs = pipe.wait_for_frames(2000); if (fs) break; }
                auto cf = fs.get_color_frame();
                cv::Mat frame(cv::Size(cf.get_width(), cf.get_height()), CV_8UC3,
                              const_cast<void*>(cf.get_data()), cv::Mat::AUTO_STEP);
                frame = frame.clone();
                pipe.stop();

                std::string label = "realsense:" + std::to_string(i);
                std::cout << "  [" << label << "] " << name << " serial=" << serial
                          << " " << frame.cols << "x" << frame.rows << std::endl;
                drawLabel(frame, label + " " + serial);
                snaps.push_back({label, frame});
            } catch (const std::exception& e) {
                std::cerr << "  [realsense:" << i << "] " << name << " serial=" << serial
                          << " ERROR: " << e.what() << std::endl;
            }
        }
    }
#else
    std::cout << "=== RealSense support not compiled (define ENABLE_REALSENSE to enable) ===" << std::endl;
#endif

    std::cout << "\nFound: " << snaps.size() << " camera(s)" << std::endl;
    if (snaps.empty()) { std::cerr << "No cameras found." << std::endl; return 1; }

    // ─── 3. 拼接成一张图 ───
    const int H = 480;
    std::vector<cv::Mat> rows;
    for (auto& s : snaps) {
        cv::Mat f = ensureBGR(s.frame);
        if (!f.isContinuous()) f = f.clone();
        if (f.rows != H) {
            double sc = static_cast<double>(H) / f.rows;
            cv::resize(f, f, cv::Size(static_cast<int>(f.cols * sc), H));
        }
        rows.push_back(f);
    }

    int totalW = 0;
    for (auto& r : rows) totalW += r.cols;
    cv::Mat canvas(H, totalW, CV_8UC3, {20, 20, 20});
    int x = 0;
    for (auto& r : rows) {
        cv::Mat roi = canvas(cv::Rect(x, 0, r.cols, r.rows));
        if (r.type() == CV_8UC3 && r.size() == roi.size()) r.copyTo(roi);
        x += r.cols;
    }

    cv::imwrite(outputPath, canvas);
    std::cout << "Saved: " << outputPath << std::endl;
    return 0;
}
