"""从 ROS 图像话题读入视频流。

默认订阅标准相机/RealSense 节点输出的彩色图 + CameraInfo。
- use_compressed=true 时改订 CompressedImage（省带宽）
- use_depth=true 时同步订阅对齐深度图（携带 RGB-D 中心距离）

发布：
  /kfs_tracker/detection       kfs_tracker/msg/KFSDetection
  /kfs_tracker/annotated_image sensor_msgs/Image（标注图）
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('image_topic', default_value='/camera/color/image_raw',
                              description='彩色图话题（raw Image 或 CompressedImage）'),
        DeclareLaunchArgument('camera_info_topic', default_value='/camera/color/camera_info',
                              description='相机内参话题；无则用兜底内参'),
        DeclareLaunchArgument('depth_topic',
                              default_value='/camera/aligned_depth_to_color/image_raw',
                              description='对齐深度图话题（仅 use_depth=true 时使用）'),
        DeclareLaunchArgument('use_compressed', default_value='false',
                              description='true: 订阅 CompressedImage；false: 订阅 raw Image'),
        DeclareLaunchArgument('use_depth', default_value='false',
                              description='true: 同步订阅 depth（RGB-D，携带中心距离）'),
        DeclareLaunchArgument('model_type', default_value='heatmap',
                              description='heatmap（角点）| bbox'),
        DeclareLaunchArgument('display_ui', default_value='false',
                              description='true: 本地 cv::imshow 显示标注图'),
        DeclareLaunchArgument('publish_annotated', default_value='true',
                              description='true: 发布标注图到 /kfs_tracker/annotated_image'),
        Node(
            package='kfs_tracker',
            executable='kfs_tracker_node',
            name='kfs_tracker_node',
            output='screen',
            parameters=[
                # 所有参数从 yaml 读取，不在 launch 中覆盖
                os.path.join(
                    get_package_share_directory('kfs_tracker'),
                    'config', 'kfs_tracker_params.yaml'),
            ],
        ),
    ])
