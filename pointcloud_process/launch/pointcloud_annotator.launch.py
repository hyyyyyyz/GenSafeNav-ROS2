#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='pointcloud_process',
            executable='pointcloud_annotator',
            name='pointcloud_annotator_node',
            output='screen',
            parameters=[{
                'input_cloud_topic':  '/livox/lidar/pointcloud',
                'tracks_topic':       '/tracked_objects_json',
                'output_cloud_topic': '/processed_pointcloud',
                'pedestrian_radius':  0.6,   # metres (half of SORT box_size=1.2)
                'cloud_queue_size':   5,
                'tracks_queue_size':  10,
            }]
        )
    ])
