"""本地直连 Intel RealSense（RGB-D）。

要求本包以 -DENABLE_REALSENSE=ON 编译、librealsense2 已安装且设备已连接。
RealSense provider 从设备 SDK 自动获取内参与对齐深度，无需 intrinsics_path。

注意：若不想在本包编译 RealSense SDK，可改用 realsense2_camera 节点发布 topic，
再用 tracker_topic.launch.py（use_depth=true）订阅，本包无需 ENABLE_REALSENSE。

发布：
  /kfs_tracker/detection       kfs_tracker/msg/KFSDetection（携带 center_distance_m）
  /kfs_tracker/annotated_image sensor_msgs/Image
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('frame_width', default_value='1280'),
        DeclareLaunchArgument('frame_height', default_value='720'),
        DeclareLaunchArgument('grab_fps', default_value='30.0'),
        DeclareLaunchArgument('model_type', default_value='heatmap',
                              description='heatmap（角点）| bbox'),
        DeclareLaunchArgument('display_ui', default_value='true'),
        DeclareLaunchArgument('publish_annotated', default_value='true'),
        Node(
            package='kfs_tracker',
            executable='kfs_tracker_node',
            name='kfs_tracker_node',
            output='screen',
            parameters=[{
                'input_source': 'local',
                'camera_type': 'realsense',
                'frame_width': LaunchConfiguration('frame_width'),
                'frame_height': LaunchConfiguration('frame_height'),
                'grab_fps': LaunchConfiguration('grab_fps'),
                'model_type': LaunchConfiguration('model_type'),
                'display_ui': LaunchConfiguration('display_ui'),
                'publish_annotated': LaunchConfiguration('publish_annotated'),
                # RealSense provider 从设备读内参，无需 intrinsics_path
            }],
        ),
    ])
