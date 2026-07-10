"""本地直连摄像头（OpenCV 或视频文件）。

复用 CameraProvider：camera_type=opencv 打开本地摄像头索引；
camera_type=file 回放视频文件（需同时提供 intrinsics_path）。

发布：
  /kfs_tracker/detection       kfs_tracker/msg/KFSDetection
  /kfs_tracker/annotated_image sensor_msgs/Image（publish_annotated=true 时）
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('camera_type', default_value='opencv',
                              description='opencv | file'),
        DeclareLaunchArgument('camera_index', default_value='0',
                              description='OpenCV 摄像头设备索引'),
        DeclareLaunchArgument('video_path', default_value='',
                              description='视频文件/流路径（camera_type=file 时）'),
        DeclareLaunchArgument('intrinsics_path', default_value='',
                              description='相机内参 YAML（OpenCV 标定格式）'),
        DeclareLaunchArgument('frame_width', default_value='1280'),
        DeclareLaunchArgument('frame_height', default_value='720'),
        DeclareLaunchArgument('grab_fps', default_value='30.0',
                              description='本地抓帧频率（Hz）'),
        DeclareLaunchArgument('model_type', default_value='heatmap',
                              description='heatmap（角点）| bbox'),
        DeclareLaunchArgument('display_ui', default_value='true',
                              description='true: 本地 cv::imshow（调试用）'),
        DeclareLaunchArgument('publish_annotated', default_value='true',
                              description='true: 发布标注图到 /kfs_tracker/annotated_image'),
        DeclareLaunchArgument('presence_service', default_value='/kfs_tracker/check_presence',
                              description='KFS 近点存在性检测 service 名'),
        Node(
            package='kfs_tracker',
            executable='kfs_tracker_node',
            name='kfs_tracker_node',
            output='screen',
            parameters=[{
                'input_source': 'local',
                'camera_type': LaunchConfiguration('camera_type'),
                'camera_index': LaunchConfiguration('camera_index'),
                'video_path': LaunchConfiguration('video_path'),
                'intrinsics_path': LaunchConfiguration('intrinsics_path'),
                'frame_width': LaunchConfiguration('frame_width'),
                'frame_height': LaunchConfiguration('frame_height'),
                'grab_fps': LaunchConfiguration('grab_fps'),
                'model_type': LaunchConfiguration('model_type'),
                'display_ui': LaunchConfiguration('display_ui'),
                'publish_annotated': LaunchConfiguration('publish_annotated'),
                'presence_service': LaunchConfiguration('presence_service'),
            }],
        ),
    ])
