cmake_minimum_required(VERSION 3.10)
project(MyCppProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TORCH_CMAKE_DIR "${CMAKE_SOURCE_DIR}/uv/Lib/site-packages/torch/share/cmake")
if(EXISTS "${TORCH_CMAKE_DIR}")
    list(PREPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_DIR}")
endif()

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(myapp
    src/main.cpp
    src/tracker.cpp
    src/2D_tracker.cpp
    src/bbox_tracking_classify.cpp
    src/heatmap_classifier_infer.cpp
)
target_link_libraries(myapp ${OpenCV_LIBS} ${TORCH_LIBRARIES})
target_compile_definitions(myapp PRIVATE -D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS)
