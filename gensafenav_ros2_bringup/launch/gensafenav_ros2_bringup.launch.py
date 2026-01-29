import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Get package directories
    bringup_dir = get_package_share_directory('gensafenav_ros2_bringup')
    livox_dir = get_package_share_directory('livox_ros_driver2')
    fast_lio_dir = get_package_share_directory('fast_lio')

    # Configuration files
    dr_spaam_config = os.path.join(bringup_dir, 'config', 'dr_spaam_params.yaml')
    sort_tracker_config = os.path.join(bringup_dir, 'config', 'sort_tracker_params.yaml')
    pointcloud_config = os.path.join(bringup_dir, 'config', 'pointcloud_to_laserscan_params.yaml')
    livox_config_path = os.path.join(livox_dir, 'config', 'MID360_config.json')
    fast_lio_config = os.path.join(fast_lio_dir, 'config', 'mid360.yaml')

    # Livox ROS2 parameters
    livox_ros2_params = [
        {"xfer_format": 4},        # 0-PointCloud2(PointXYZRTL)
        {"multi_topic": 0},        # 0-All LiDARs share the same topic
        {"data_src": 0},           # 0-lidar
        {"publish_freq": 10.0},    # 10 Hz
        {"output_data_type": 0},
        {"frame_id": 'livox_frame'},
        {"lvx_file_path": '/home/livox/livox_test.lvx'},
        {"user_config_path": livox_config_path},
        {"cmdline_input_bd_code": 'livox0000000001'}
    ]

    # Launch arguments
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Launch RViz for visualization'
    )

    rviz_config_arg = DeclareLaunchArgument(
        'rviz_config',
        default_value=os.path.join(bringup_dir, 'rviz', 'gensafenav.rviz'),
        description='Path to RViz config file'
    )

    use_livox_arg = DeclareLaunchArgument(
        'use_livox',
        default_value='true',
        description='Launch Livox driver (set false if using rosbag or other lidar)'
    )

    # ====================
    # 0. Livox MID360 Driver (Optional)
    # ====================
    livox_driver_node = Node(
        package='livox_ros_driver2',
        executable='livox_ros_driver2_node',
        name='livox_lidar_publisher',
        output='screen',
        parameters=livox_ros2_params,
        condition=IfCondition(LaunchConfiguration('use_livox'))
    )

    # ====================
    # 1. FAST-LIO (Odometry and Mapping)
    # ====================
    fast_lio_node = Node(
        package='fast_lio',
        executable='fastlio_mapping',
        name='fastlio_mapping',
        parameters=[fast_lio_config],
        output='screen'
    )

    # ====================
    # 2. PointCloud to LaserScan Converter
    # ====================
    pointcloud_to_laserscan_node = Node(
        package='pointcloud_to_laserscan',
        executable='pointcloud_to_laserscan_node',
        name='pointcloud_to_laserscan',
        parameters=[pointcloud_config],
        remappings=[
            ('cloud_in', '/livox/lidar'),
            ('scan', '/scan')
        ],
        output='screen'
    )

    # ====================
    # 3. DR-SPAAM Person Detector
    # ====================
    dr_spaam_node = Node(
        package='dr_spaam_ros2',
        executable='dr_spaam_w_score_ros',
        name='dr_spaam_ros2',
        parameters=[dr_spaam_config],
        output='screen'
    )

    # ====================
    # 4. SORT Tracker
    # ====================
    sort_tracker_node = Node(
        package='sort_tracker',
        executable='sort_tracker',
        name='sort_tracker_node',
        parameters=[sort_tracker_config],
        output='screen'
    )

    # ====================
    # 5. Trajectory Predictor
    # ====================
    predictor_node = Node(
        package='predictor',
        executable='predictor',
        name='predictor_node',
        output='screen'
    )

    # ====================
    # RViz Visualization
    # ====================
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', LaunchConfiguration('rviz_config')],
        condition=IfCondition(LaunchConfiguration('use_rviz')),
        output='screen'
    )

    return LaunchDescription([
        # Launch arguments
        use_rviz_arg,
        rviz_config_arg,
        use_livox_arg,

        # Nodes with sequential delays
        # 0. Start Livox first
        livox_driver_node,

        # 1. Wait 2s for Livox to initialize, then start FAST-LIO
        TimerAction(
            period=2.0,
            actions=[fast_lio_node]
        ),

        # 2. Wait 3s, then start pointcloud converter
        TimerAction(
            period=3.0,
            actions=[pointcloud_to_laserscan_node]
        ),

        # 3. Wait 4s, then start DR-SPAAM detector
        TimerAction(
            period=4.0,
            actions=[dr_spaam_node]
        ),

        # 4. Wait 5s, then start SORT tracker
        TimerAction(
            period=5.0,
            actions=[sort_tracker_node]
        ),

        # 5. Wait 6s, then start predictor
        TimerAction(
            period=6.0,
            actions=[predictor_node]
        ),

        # 6. Start RViz immediately (optional)
        rviz_node,
    ])
