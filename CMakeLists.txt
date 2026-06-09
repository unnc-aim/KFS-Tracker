cmake_minimum_required(VERSION 3.8)
project(kfs_tracker)

# C++17（与现有源码一致）
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(APPLE)
  set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "macOS architecture")
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# --- 可选 RealSense 支持 ---
option(ENABLE_REALSENSE "Build with Intel RealSense camera support" OFF)

# ==================== ament / ROS 依赖 ====================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# ==================== LibTorch（沿用：CMAKE_CURRENT_SOURCE_DIR 在 colcon 下指向包根）====================
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/deps/libtorch/share/cmake")
find_package(Torch REQUIRED)

# ==================== OpenCV ====================
find_package(OpenCV REQUIRED)

# ==================== RealSense（条件）====================
if(ENABLE_REALSENSE)
  find_package(realsense2 REQUIRED)
endif()

# ==================== 自定义消息 ====================
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/KFSDetection.msg"
  DEPENDENCIES std_msgs geometry_msgs
)
ament_export_dependencies(rosidl_default_runtime)

# ==================== kfs_tracker_core 静态库（CLI 与节点共用）====================
set(KFS_CORE_SOURCES
  src/tracker.cpp
  src/2D_tracker.cpp
  src/bbox_tracking_classify.cpp
  src/heatmap_classifier_infer.cpp
  src/heatmap_tracking_classify.cpp
  src/camera_provider.cpp
  src/camera_provider_opencv.cpp
  src/infer_pipeline.cpp
)
if(ENABLE_REALSENSE)
  list(APPEND KFS_CORE_SOURCES src/camera_provider_realsense.cpp)
endif()

add_library(kfs_tracker_core STATIC ${KFS_CORE_SOURCES})
target_include_directories(kfs_tracker_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${OpenCV_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(kfs_tracker_core PUBLIC
  ${OpenCV_LIBS}
  ${TORCH_LIBRARIES}
)
target_compile_definitions(kfs_tracker_core PUBLIC _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS)
if(ENABLE_REALSENSE)
  target_link_libraries(kfs_tracker_core PUBLIC ${realsense2_LIBRARY})
  # PUBLIC：core 内 camera_provider.cpp 的 #ifdef ENABLE_REALSENSE 分发依赖此宏
  target_compile_definitions(kfs_tracker_core PUBLIC ENABLE_REALSENSE)
endif()

# ==================== CLI 可执行（原 kfs_tracker，无 ROS 依赖）====================
add_executable(kfs_tracker src/main.cpp)
target_link_libraries(kfs_tracker PRIVATE kfs_tracker_core)

# ==================== ROS 节点可执行 ====================
add_executable(kfs_tracker_node src/kfs_tracker_node.cpp)
target_link_libraries(kfs_tracker_node PRIVATE kfs_tracker_core)
ament_target_dependencies(kfs_tracker_node
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  cv_bridge
  message_filters
  ament_index_cpp
)
# 消费本包自定义消息：用 typesupport target 链接（Humble 现代写法，
# 不要把 ${PROJECT_NAME} 放进 ament_target_dependencies）
rosidl_get_typesupport_target(kfs_typesupport_target
  ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(kfs_tracker_node PRIVATE ${kfs_typesupport_target})
add_dependencies(kfs_tracker_node ${${PROJECT_NAME}_EXPORTED_TARGETS})

# ==================== 安装 ====================
install(TARGETS kfs_tracker kfs_tracker_node
  DESTINATION lib/${PROJECT_NAME})

# launch / config 整目录安装
install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME})

# 模型只装 .pt 文件（避免 .DS_Store 等杂项）
install(DIRECTORY model
  DESTINATION share/${PROJECT_NAME}
  FILES_MATCHING PATTERN "*.pt")

# ==================== 测试（统一跳过 copyright/cpplint lint）====================
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
