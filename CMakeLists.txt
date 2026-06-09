cmake_minimum_required(VERSION 3.16)
project(KFS-Tracker)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Platform-specific settings ---
if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "macOS architecture")
endif()

# --- Optional RealSense support ---
option(ENABLE_REALSENSE "Build with Intel RealSense camera support" OFF)

# --- LibTorch (C++ standalone, from deps/) ---
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/deps/libtorch/share/cmake")
find_package(Torch REQUIRED)

# --- OpenCV ---
find_package(OpenCV REQUIRED)

# --- RealSense (conditional) ---
if(ENABLE_REALSENSE)
    find_package(realsense2 REQUIRED)
endif()

set(KFS_SOURCES
    src/main.cpp
    src/tracker.cpp
    src/2D_tracker.cpp
    src/bbox_tracking_classify.cpp
    src/heatmap_classifier_infer.cpp
    src/heatmap_tracking_classify.cpp
    src/camera_provider.cpp
    src/camera_provider_opencv.cpp
)

if(ENABLE_REALSENSE)
    list(APPEND KFS_SOURCES src/camera_provider_realsense.cpp)
endif()

add_executable(kfs_tracker ${KFS_SOURCES})

target_include_directories(kfs_tracker PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(kfs_tracker PRIVATE
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
)

if(ENABLE_REALSENSE)
    target_link_libraries(kfs_tracker PRIVATE ${realsense2_LIBRARY})
    target_compile_definitions(kfs_tracker PRIVATE ENABLE_REALSENSE)
endif()

target_compile_definitions(kfs_tracker PRIVATE
    -D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS
)
