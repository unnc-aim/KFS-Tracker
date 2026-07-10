# KFS-Tracker

Robocon 场景下的 KFS 目标追踪系统。基于 C++17 + OpenCV + LibTorch 实现，支持摄像头实时推理与单图模式。

## 功能概览

| 模块 | 功能 |
|------|------|
| **BboxTrackingClassifier** | 基于 LibTorch 的目标检测（Bbox 定位，EfficientNet-B3 回归头） |
| **HeatmapTrackingClassifier** | 基于 U-Net 的角点热力图检测（4 通道高斯热力图 → argmax 解码角点，单体式含可视化） |
| **ImageClassifierInfer** | 独立图像分类（`tracker_classifier.pt`），统一负责 bbox / heatmap 模式的分类输出 |
| **HeatmapCornerInfer** | 拆分式角点热力图推理（仅输出 4 个角点，配合 `PlaneTracker2D` 做位姿估计） |
| **CubeTracker3D** | 立方体 3D 位姿估计：颜色分割 → 六边形轮廓拟合 → 消失点分析 → 6 点 PnP |
| **PlaneTracker2D** | 平面四边形 2D 位姿估计：颜色分割 → 四边形检测 → solvePnP |
| **CameraProvider** | 统一相机抽象层，支持 OpenCV 摄像头 / RealSense / 标定文件三种模式 |

## 项目结构

```
KFS-Tracker/
├── CMakeLists.txt              # 构建配置（C++17, LibTorch, OpenCV, 可选 RealSense）
├── config/
│   └── intrinsics_example.yaml # 相机内参示例（OpenCV 标定格式）
├── deps/
│   └── libtorch/               # LibTorch C++ 发行版（需自行下载，见下方说明）
├── doc/
│   ├── pose_estimation_requirements.md    # 3D 立方体位姿估计需求文档
│   └── corner_point_heatmap_requirements.md # 角点热力图检测需求文档
├── model/
│   ├── tracker_localizer.pt        # Bbox 回归模型（TorchScript, ~45MB）
│   ├── tracker_classifier.pt       # 图像分类模型（TorchScript, ~16MB；统一负责分类）
│   ├── corner_heatmap_localizer.pt # 角点热力图模型（TorchScript, ~51MB）
│   └── corner_point_localizer.pt   # 角点定位模型（TorchScript, ~45MB）
├── src/
│   ├── main.cpp                # 入口：CLI 解析、CameraProvider 初始化、推理循环
│   ├── bbox_tracking_classify.h/cpp  # Bbox 检测推理
│   ├── heatmap_tracking_classify.h/cpp # 热力图角点检测推理（单体式，含可视化）
│   ├── heatmap_classifier_infer.h/cpp # 拆分式推理：HeatmapCornerInfer + ImageClassifierInfer
│   ├── tracker.h/cpp           # CubeTracker3D 实现
│   ├── 2D_tracker.h/cpp        # PlaneTracker2D 实现（2D 位姿估计）
│   ├── camera_provider.h       # 相机抽象接口 + Intrinsics/FrameBundle 定义
│   ├── camera_provider.cpp     # 工厂函数 + 内参加载
│   ├── camera_provider_opencv.cpp  # OpenCV 摄像头 + 文件模式实现
│   ├── camera_provider_realsense.cpp # RealSense 实现（条件编译）
│   └── model_code/             # Python 模型训练代码
│       ├── tracker_model.py / tracker_train.py / tracker_dataset.py
│       ├── heatmap_model.py / heatmap_train.py / heatmap_dataset.py
│       ├── classifier_model.py / classifier_train.py / classifier_dataset.py
│       ├── corner_point_train.py / corner_point_dataset.py
│       └── sample_test_images.py
└── dataset-generator/          # 数据集生成工具（独立子项目）
```

## 环境依赖

| 依赖 | 版本 | 说明 |
|------|------|------|
| CMake | ≥ 3.16 | 构建系统 |
| C++17 编译器 | Apple Clang 17+ / GCC 9+ | |
| OpenCV | 4.x | 图像处理、相机采集 |
| LibTorch (C++) | 2.7.0+ | 模型推理（CPU 版本即可） |
| Intel RealSense SDK | 2.x | 可选，用于深度相机支持 |

### macOS 安装依赖

```bash
# OpenCV
brew install opencv

# LibTorch（下载 CPU 版本并解压到 deps/）
mkdir -p deps
curl -L https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.7.0.zip -o /tmp/libtorch.zip
unzip -qo /tmp/libtorch.zip -d deps/
```

### Linux (x86_64) 安装依赖

```bash
# OpenCV
sudo apt install libopencv-dev

# LibTorch
mkdir -p deps
curl -L https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.7.0%2Bcpu.zip -o /tmp/libtorch.zip
unzip -qo /tmp/libtorch.zip -d deps/
```

## 构建

```bash
# 标准构建（不含 RealSense）
mkdir build && cd build
cmake ..
cmake --build .

# 启用 RealSense 支持（需要已安装 librealsense2）
cmake .. -DENABLE_REALSENSE=ON
cmake --build .
```

## 使用方法

### 摄像头实时推理

```bash
# 默认热力图角点检测模式（等效 --model model/corner_heatmap_localizer.pt --model_type heatmap）
./build/kfs_tracker --camera_type opencv

# Bbox 检测模式
./build/kfs_tracker --camera_type opencv --model model/tracker_localizer.pt --model_type bbox

# 指定分类模型
./build/kfs_tracker --camera_type opencv --classifier_model model/tracker_classifier.pt

# 指定设备索引和分辨率
./build/kfs_tracker --camera_type opencv --camera_index 1 --width 640 --height 480

# 携带标定文件
./build/kfs_tracker --camera_type opencv --intrinsics config/intrinsics_example.yaml

# RealSense 深度相机（需要 ENABLE_REALSENSE=ON 编译）
./build/kfs_tracker --camera_type realsense

# 文件模式：已知标定 + 视频文件
./build/kfs_tracker --camera_type file --video test.mp4 --intrinsics config/intrinsics_example.yaml
```

### 单图推理

```bash
# 默认热力图角点检测模式
./build/kfs_tracker --image input.jpg --output annotated.jpg

# Bbox 模式
./build/kfs_tracker --image input.jpg --output annotated.jpg \
    --model model/tracker_localizer.pt --model_type bbox
```

### 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `DISABLE_RENDERER` | 未设置 | 设为 `1` / `true` 可禁止弹窗渲染标注结果 |

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--camera_type` | — | 相机类型：`opencv` / `realsense` / `file`；设置后进入流模式 |
| `--camera_index` | `0` | OpenCV 摄像头设备索引 |
| `--intrinsics` | — | 相机内参 YAML 文件路径（OpenCV 标定格式） |
| `--video` | — | 视频文件路径（`--camera_type file` 模式） |
| `--width` | `1280` | 期望采集宽度 |
| `--height` | `720` | 期望采集高度 |
| `--model_type` | `heatmap` | 模型类型：`heatmap`（角点热力图检测）/ `bbox`（Bbox 检测） |
| `--image` | `image.jpg` | 单图模式输入路径 |
| `--output` | `annotated.jpg` | 单图模式输出路径 |
| `--model` | 跟随 `--model_type` | 检测/角点模型路径（heatmap→`corner_heatmap_localizer.pt`，bbox→`tracker_localizer.pt`） |
| `--classifier_model` | `model/tracker_classifier.pt` | 分类模型路径（bbox / heatmap 模式统一使用） |
| `--input_size` | `300` | 模型输入尺寸 |

## 相机内参标定

相机内参以 [OpenCV 标准标定格式](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html) 的 YAML 文件提供：

```yaml
%YAML:1.0
---
camera_matrix: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ fx, 0.0, cx,
           0.0, fy, cy,
           0.0, 0.0, 1.0 ]
distortion_coefficients: !!opencv-matrix
   rows: 1
   cols: 5
   dt: d
   data: [ k1, k2, p1, p2, k3 ]
```

- **RealSense 模式**：内参由设备 SDK 自动获取，无需 YAML 文件
- **OpenCV / File 模式**：不提供 `--intrinsics` 时使用 identity 兜底（会打印警告，3D 位姿估计不可信赖）

## 输出示例

### Bbox 模式（`--model_type bbox`）

流模式终端输出（推理耗时 + 理论帧率）：

```
[frame 0] infer: 8.3ms (120.5 fps) | bbox: [100, 200, 300 x 200]
  class: T_05 (id=4, p=0.920)
[frame 1] infer: 7.9ms (126.6 fps) | bbox: [105, 198, 298 x 196]
  class: T_05 (id=4, p=0.910)
```

渲染窗口叠加层显示：
- 左上角 `class: ...` — 分类结果（由 `tracker_classifier.pt` 统一给出）
- `Infer: X.X ms` — 当前帧推理耗时
- `Max FPS: X.X (EMA X.X)` — 理论最高帧率 + 指数移动平均

### 热力图模式（`--model_type heatmap`，默认）

流模式终端输出（角点 + 由 `tracker_classifier.pt` 给出的分类）：

```
[frame 0] infer: 12.5ms (80.0 fps) | corners: (320.5,180.2) (315.8,520.7) (790.3,515.1) (795.1,175.8)
  class: T_05 (id=4, p=0.901)
```

渲染窗口叠加层显示：
- 绿色四边形连线 + 4 个角点圆圈（红=TL、蓝=BL、青=BR、黄=TR）
- `Heatmap: X.X ms` — 推理耗时
- 左上角 `class: ...` — 分类结果（由 `tracker_classifier.pt` 统一给出）

> **说明**：热力图模型本身只输出 4 个角点坐标；分类由独立的 `ImageClassifierInfer(tracker_classifier.pt)` 统一负责，与 bbox 模式共用同一分类模型。

## 作为 ROS 2 节点封装（ROS 2 Humble / colcon）

本包同时是一个 ROS 2 Humble 包 `kfs_tracker`，提供节点 `kfs_tracker_node`，支持「从 ROS 图像话题读入视频流」与「本地直连摄像头」两种输入，并把标注图与 tracker + classification 结果发布到话题。核心算法复用 `kfs_tracker_core` 静态库，与 CLI 可执行 `kfs_tracker` 并存（无 ROS 环境下 CLI 仍可独立运行）。

> 部署约束：ROS 2 构建与运行须在 **Linux（Ubuntu 22.04）/ ROS 2 Humble** 上；macOS 无 ROS 2 Humble 官方支持，不能 `colcon` 构建。

### ROS 2 环境依赖安装（Ubuntu 22.04 / ROS 2 Humble）

假设已安装 ROS 2 Humble 并 source 过 `/opt/ros/humble/setup.bash`。本节点额外需要 **cv_bridge + 系统 OpenCV** 与 **LibTorch**。

1. **系统 OpenCV + cv_bridge / message_filters / image_transport**（apt 安装）：

   ```bash
   sudo apt update
   sudo apt install -y \
       ros-humble-cv-bridge \
       ros-humble-message-filters \
       ros-humble-image-transport \
       libopencv-dev
   ```

   > ⚠️ **ABI 一致性（最关键）**：cv_bridge 链接的是 apt 的 OpenCV 4.x。**LibTorch 必须使用同一份系统 OpenCV**（即下载「不带 bundled OpenCV」的 CPU 版 libtorch，让它复用系统的 `libopencv_*`）。否则同一进程加载两份 OpenCV 会运行期 double-free / `imshow` 崩溃。验证：构建后 `ldd install/kfs_tracker/lib/kfs_tracker/kfs_tracker_node | grep -i opencv`，应只看到一份 `libopencv_*.so.4.x`。

2. **LibTorch（C++，cxx11 ABI CPU 版）**——解压到包内 `deps/libtorch`（`CMakeLists.txt` 固定从 `${CMAKE_CURRENT_SOURCE_DIR}/deps/libtorch/share/cmake` 加载）：

   ```bash
   cd <本包根目录 KFS-Tracker>
   mkdir -p deps
   # Ubuntu 22.04 / gcc 9+ 必须用 cxx11 ABI 版（文件名含 cxx11-abi）
   curl -L https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.7.0%2Bcpu.zip -o /tmp/libtorch.zip
   unzip -qo /tmp/libtorch.zip -d deps/
   ```

   > ⚠️ 务必选 **`cxx11-abi`** 版；误用 pre-cxx11 版会链接报 `undefined reference to c10::...` 或 `std::...`。
   > 验证：`ls deps/libtorch/share/cmake/Torch/TorchConfig.cmake` 应存在。

3. **（可选）Intel RealSense SDK**——仅当用 `camera_type=realsense` **本地直连**时需要；若改用 `realsense2_camera` 节点发布 topic 再用 `tracker_topic.launch.py` 订阅，则**不需要**：

   ```bash
   sudo apt install -y librealsense2-dev
   # 之后用 -DENABLE_REALSENSE=ON 构建（见下）
   ```

### 构建

```bash
cd ~/26RC_R2_ws
# 默认（不含 RealSense 本地直连）
colcon build --packages-select kfs_tracker
# 启用 RealSense 本地直连（需要 librealsense2）
colcon build --packages-select kfs_tracker --cmake-args -DENABLE_REALSENSE=ON

source install/setup.bash
```

构建产物：
- `kfs_tracker`（CLI，原单图/流模式）
- `kfs_tracker_node`（ROS 2 节点）

### 发布 / 订阅话题

| 话题 | 类型 | 方向 | 说明 |
|------|------|------|------|
| `/kfs_tracker/detection` | `kfs_tracker/msg/KFSDetection` | 发布 | 定位（角点/bbox）+ 分类 + 推理耗时 + 可选中心距离 |
| `/kfs_tracker/annotated_image` | `sensor_msgs/Image` | 发布 | 标注图（OpenCV UI 画面） |
| `image_topic` | `sensor_msgs/Image` 或 `CompressedImage` | 订阅 | topic 模式彩色图输入 |
| `camera_info_topic` | `sensor_msgs/CameraInfo` | 订阅 | 相机内参（topic 模式，可选） |
| `depth_topic` | `sensor_msgs/Image`（16UC1 mm） | 订阅 | 对齐深度图（`use_depth=true` 时） |

`KFSDetection` 字段：`header` / `valid` / `model_type`（heatmap\|bbox）/ `corners[4]` / `bbox_min`·`bbox_max` / `class_id`·`class_name`·`class_score`·`class_valid` / `infer_ms` / `center_distance_m`（-1.0=无深度）。

### KFS 近点存在性 Service

节点额外提供一个按需调用的 service：`/kfs_tracker/check_presence`，类型为 `kfs_tracker/srv/CheckKFSPresence`。它用于判断当前相机画面中是否存在“足够近”的 KFS 方块。

原理：
1. service 调用时读取节点缓存的最新一帧彩色图、深度图和相机内参。
2. 使用 `corner_heatmap_localizer.pt` 对彩色图做角点热力图追踪，得到 KFS 平面 4 个角点。
3. 使用 `PlaneTracker2D::detectFromCorners` 根据角点和相机内参计算位姿，并得到重投影误差。
4. 如果重投影误差无效或大于阈值，认为角点/位姿不可靠，返回 `has_kfs_cube=false`。
5. 如果重投影误差合格，则在角点四边形内部统计 16UC1 深度图的有效深度均值。
6. 平均距离落在 `kfs_presence_detector.cpp` 顶部的近点距离区间内时返回 `true`，否则返回 `false`。

> 近点距离区间目前写在 `src/kfs_presence_detector.cpp` 顶部：`kNearestPointMinDistanceM` / `kNearestPointMaxDistanceM`，后续按实测距离直接改这两个常量。

调用方法：

```bash
# 先确保节点已经启动，并且已有图像输入。
# topic + RGB-D 路径示例：
ros2 launch kfs_tracker tracker_topic.launch.py use_depth:=true

# 查看 service 类型
ros2 service type /kfs_tracker/check_presence

# 调用存在性判断（请求为空）
ros2 service call /kfs_tracker/check_presence kfs_tracker/srv/CheckKFSPresence "{}"
```

返回字段：

| 字段 | 说明 |
|------|------|
| `has_kfs_cube` | 最终判断结果：是否存在满足重投影和近点距离条件的 KFS 方块 |
| `frame_available` | 节点是否已经收到至少一帧图像 |
| `heatmap_valid` | 热力图角点追踪是否有效 |
| `pose_solved` | `PlaneTracker2D` 是否得到低误差位姿 |
| `depth_valid` | 追踪平面内是否有可用深度样本 |
| `reprojection_error` | 角点位姿重投影误差，单位 px；`-1` 表示不可用 |
| `average_plane_distance_m` | 追踪四边形内部有效深度均值，单位 m；`-1` 表示不可用 |
| `valid_depth_samples` | 参与平均距离计算的有效深度像素数 |
| `message` | 对当前判断路径的文字说明 |

示例返回：

```yaml
has_kfs_cube: true
frame_available: true
heatmap_valid: true
pose_solved: true
depth_valid: true
reprojection_error: 3.42
average_plane_distance_m: 0.84
valid_depth_samples: 12540
message: kfs cube present
```

### 节点参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `input_source` | `topic` | `local`（本地直连）\| `topic`（订阅 ROS 图像） |
| `model_type` | `heatmap` | `heatmap`（角点）\| `bbox` |
| `model` / `classifier_model` / `presence_model` | share 下默认 | 模型路径，`presence_model` 固定用于 service 的热力图角点追踪 |
| `input_size` | `300` | 模型输入尺寸 |
| `display_ui` | `false` | `true` 时本地 `cv::imshow`（关闭 UI 渲染则保持 `false`，headless 必须关） |
| `publish_annotated` | `true` | 是否发布标注图话题 |
| `image_topic` | `/camera/color/image_raw` | 彩色图话题 |
| `camera_info_topic` | `/camera/color/camera_info` | 内参话题 |
| `depth_topic` | `/camera/aligned_depth_to_color/image_raw` | 对齐深度图话题 |
| `use_compressed` | `false` | `true` 订 `CompressedImage`，否则 raw `Image` |
| `use_depth` | `false` | `true` 同步订深度（RGB-D，携带 `center_distance_m`） |
| `camera_type` | `opencv` | `opencv`\|`realsense`\|`file`（local 模式） |
| `camera_index` / `video_path` / `intrinsics_path` | — | local 模式参数 |
| `frame_width` / `frame_height` / `grab_fps` | 1280 / 720 / 30.0 | local 模式采集 |
| `detection_topic` / `annotated_topic` | 见上 | 输出话题名（绝对名） |
| `presence_service` | `/kfs_tracker/check_presence` | KFS 近点存在性判断 service 名 |

### Launch 文件速查

| Launch | 输入 | 典型场景 |
|--------|------|----------|
| `tracker_local.launch.py` | OpenCV 摄像头 / 视频文件 | 笔记本摄像头调试、回放离线视频 |
| `tracker_realsense.launch.py` | 直连 Intel RealSense RGB-D | 需 `ENABLE_REALSENSE=ON` 编译且设备已连接 |
| `tracker_topic.launch.py` | 订阅 ROS 图像话题 | 已有相机节点发布图像流（最通用、部署首选） |

三者均发布 `/kfs_tracker/detection`（`KFSDetection`）与 `/kfs_tracker/annotated_image`（标注图）。

### 启动示例

```bash
# 1) 从 ROS 图像话题读入（raw Image + CameraInfo）
ros2 launch kfs_tracker tracker_topic.launch.py \
    image_topic:=/image_raw model_type:=heatmap

# 2) 压缩图输入
ros2 launch kfs_tracker tracker_topic.launch.py \
    image_topic:=/camera/image_raw/compressed use_compressed:=true

# 3) RGB-D（RealSense2 节点发布 color + aligned_depth，携带中心距离）
ros2 launch kfs_tracker tracker_topic.launch.py \
    use_depth:=true image_topic:=/camera/color/image_raw \
    depth_topic:=/camera/aligned_depth_to_color/image_raw \
    camera_info_topic:=/camera/color/camera_info

# 4) 本地直连 OpenCV 摄像头（调试，开 imshow）
ros2 launch kfs_tracker tracker_local.launch.py camera_type:=opencv display_ui:=true

# 5) 回放视频文件（file 模式需内参）
ros2 launch kfs_tracker tracker_local.launch.py camera_type:=file \
    video_path:=/path/test.mp4 intrinsics_path:=/path/config/intrinsics_example.yaml

# 6) 本地直连 RealSense（需 ENABLE_REALSENSE=ON 构建且设备已连）
ros2 launch kfs_tracker tracker_realsense.launch.py
```

### RealSense 深度的两种路径

无论哪种路径，深度都是用来填充 `KFSDetection.center_distance_m`（中心像素的 16UC1 mm 深度 ÷ 1000）；不开深度时该字段恒为 `-1.0`。

> ⚠️ **关键**：`tracker_topic.launch.py` **默认 `use_depth:=false`，此时没有深度数据**（`center_distance_m=-1.0`）。要拿 RealSense 深度须显式开 `use_depth:=true`。

| | `tracker_realsense.launch.py` | `tracker_topic.launch.py` + `use_depth:=true` |
|------|------|------|
| RealSense 接入 | 本包 `librealsense2` SDK **直连**设备 | 订阅 `realsense2_camera` 节点发布的 depth 话题 |
| 本包是否要 `ENABLE_REALSENSE=ON` | **要** | 不要 |
| 是否要装 RealSense SDK | 要 | 不要（由 `realsense2_camera` 节点负责） |
| 延迟 | 低（SDK 直采） | 多一跳（经 ROS 话题） |
| 内参来源 | SDK 自动获取 | `camera_info_topic` |
| 深度默认行为 | 始终带 RGB-D | **默认关**，须手动 `use_depth:=true` |

- **路径①（本地直连，延迟低）**：本包 `ENABLE_REALSENSE=ON` 编译 + librealsense2 已装 + 设备连接。
- **路径②（部署首选，无需 SDK）**：用 `realsense2_camera` 节点发布 color + aligned_depth，本包以 `tracker_topic.launch.py` + `use_depth:=true` 订阅。`depth_topic` 默认即 `/camera/aligned_depth_to_color/image_raw`，故只需：

  ```bash
  ros2 launch kfs_tracker tracker_topic.launch.py use_depth:=true
  ```

### 验证

```bash
ros2 interface show kfs_tracker/msg/KFSDetection     # 查看消息字段
ros2 interface show kfs_tracker/srv/CheckKFSPresence # 查看 service 字段
ros2 topic echo /kfs_tracker/detection               # 观察检测结果
ros2 topic hz /kfs_tracker/detection                 # 推理吞吐
ros2 service call /kfs_tracker/check_presence kfs_tracker/srv/CheckKFSPresence "{}"
rqt_image_view /kfs_tracker/annotated_image          # 观察标注图

# ABI 健康检查：应只链到一份 OpenCV
ldd install/kfs_tracker/lib/kfs_tracker/kfs_tracker_node | grep -i opencv
```

### CLI 回归（确保封装未破坏原行为）

```bash
ros2 run kfs_tracker kfs_tracker --camera_type opencv --model_type heatmap
```

---

## 许可证

本项目仅供 Robocon 比赛使用。
