import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # ── Package share directories ──────────────────────────────────────────
    livox_pkg       = get_package_share_directory('livox_ros_driver2')
    fast_lio_pkg    = get_package_share_directory('fast_lio')
    pc2ls_pkg       = get_package_share_directory('pointcloud_to_laserscan')
    pc_process_pkg  = get_package_share_directory('pointcloud_process')
    dr_spaam_pkg    = get_package_share_directory('dr_spaam_ros2')
    sort_pkg        = get_package_share_directory('sort_tracker')
    predictor_pkg   = get_package_share_directory('predictor')
    bringup_pkg     = get_package_share_directory('gensafenav_ros2_bringup')

    rviz_cfg = os.path.join(bringup_pkg, 'rviz', 'gensafenav_ros2.rviz')

    # ── Launch arguments ───────────────────────────────────────────────────
    declare_rviz = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Launch RViz2 for visualisation'
    )

    # ── 1. Livox MID360 driver ─────────────────────────────────────────────
    livox_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(livox_pkg, 'launch', 'msg_MID360_launch.py')
        )
    )

    # ── 2. FAST-LIO mapping ────────────────
    fast_lio_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(fast_lio_pkg, 'launch', 'mapping.launch.py')
        ),
        launch_arguments={
            'config_file': 'mid360.yaml',
            'rviz': 'false',
        }.items()
    )

    # ── 3. PointCloud2 → LaserScan ─────────────────────────────────────────
    pc2ls_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pc2ls_pkg, 'launch', 'pointcloud_to_laserscan.launch.py')
        )
    )

    # ── 4. DR-SPAAM person detector ────────────────────────────────────────
    dr_spaam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(dr_spaam_pkg, 'launch', 'dr_spaam_ros2.launch.py')
        )
    )

    # ── 5. SORT tracker ────────────────────────────────────────────────────
    sort_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(sort_pkg, 'launch', 'sort_tracker.launch.py')
        )
    )

    # ── 6. Trajectory predictor ────────────────────────────────────────────
    predictor_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(predictor_pkg, 'launch', 'predictor.launch.py')
        )
    )

    # ── 7. Pointcloud process ────────────────────────────────────────────
    pc_process_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pc_process_pkg, 'launch', 'pointcloud_process.launch.py')
        )
    )

    # ── 8. RViz2 ───────────────────────────────────────────────────────────
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_cfg],
        condition=IfCondition(LaunchConfiguration('use_rviz')),
        output='screen'
    )

    return LaunchDescription([
        declare_rviz,
        livox_launch,
        fast_lio_launch,
        pc2ls_launch,
        dr_spaam_launch,
        sort_launch,
        predictor_launch,
        pc_process_launch,
        rviz_node,
    ])
