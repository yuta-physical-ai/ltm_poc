#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node
import os

def generate_launch_description():
    image_path    = DeclareLaunchArgument('image_path', default_value='/home/jetson/ros_ws/src/ltm_poc/data/sample.jpg')
    publish_rate  = DeclareLaunchArgument('publish_rate', default_value='5.0')
    yolo_model    = DeclareLaunchArgument('yolo_model', default_value='yolov8n.pt')
    yolo_conf     = DeclareLaunchArgument('yolo_conf', default_value='0.25')
    imgsz         = DeclareLaunchArgument('imgsz', default_value='640')
    target_text   = DeclareLaunchArgument('target_text', default_value='red bottle')
    clip_score_th = DeclareLaunchArgument('clip_score_threshold', default_value='0.10')
    yolo_min_box  = DeclareLaunchArgument('min_box_conf', default_value='0.10')
    fov_deg       = DeclareLaunchArgument('fov_deg', default_value='60.0')
    angular_kp    = DeclareLaunchArgument('angular_kp', default_value='0.8')
    linear_speed  = DeclareLaunchArgument('linear_speed', default_value='0.35')
    conf_thresh   = DeclareLaunchArgument('conf_thresh', default_value='0.05')

    show_viewer_window  = DeclareLaunchArgument('show_viewer_window',  default_value='False')
    publish_debug_image = DeclareLaunchArgument('publish_debug_image', default_value='True')
    start_image_view    = DeclareLaunchArgument('start_image_view',    default_value='True')

    img_pub = Node(
        package='image_publisher',
        executable='image_publisher_node',
        namespace='camera',
        parameters=[{
            'filename': LaunchConfiguration('image_path'),
            'publish_rate': LaunchConfiguration('publish_rate'),
        }],
        output='screen',
        emulate_tty=True,
    )

    detector = Node(
        package='ltm_poc',
        executable='simple_detector_node',
        name='simple_detector_node',
        parameters=[{
            'input_topic': '/camera/image_raw',
            'model': LaunchConfiguration('yolo_model'),
            'conf': LaunchConfiguration('yolo_conf'),
            'imgsz': LaunchConfiguration('imgsz'),
        }],
        output='screen',
        emulate_tty=True,
    )

    vlm_clip = Node(
        package='ltm_poc',
        executable='vlm_clip_node',
        name='vlm_clip_node',
        parameters=[{
            'image_topic': '/camera/image_raw',
            'detections_topic': '/detections',
            'target_text': LaunchConfiguration('target_text'),
            'score_threshold': LaunchConfiguration('clip_score_threshold'),
            'min_box_conf': LaunchConfiguration('min_box_conf'),
            'fov_deg': LaunchConfiguration('fov_deg'),
            'publish_scores': True,
            'max_eval': 4,
        }],
        output='screen',
        emulate_tty=True,
    )

    controller = Node(
        package='ltm_poc',
        executable='controller_node',
        name='controller_node',
        parameters=[{
            'angular_kp': LaunchConfiguration('angular_kp'),
            'linear_speed': LaunchConfiguration('linear_speed'),
            'conf_thresh': LaunchConfiguration('conf_thresh'),
        }],
        remappings=[('/cmd_vel', '/turtle1/cmd_vel')],
        output='screen',
        emulate_tty=True,
    )

    turtlesim_node = Node(
        package='turtlesim',
        executable='turtlesim_node',
        name='turtlesim',
        parameters=[{'use_sim_time': False}],
        output='screen',
        emulate_tty=True,
    )

    teleport_up = ExecuteProcess(
        cmd=[
            'bash', '-lc',
            'ros2 service call /turtle1/teleport_absolute turtlesim/srv/TeleportAbsolute "{x: 5.5, y: 5.5, theta: 1.57}"'
        ],
        output='screen',
    )

    viewer = Node(
        package='ltm_poc',
        executable='viewer_node',
        name='viewer_node',
        parameters=[{
            'image_topic': '/camera/image_raw',
            'detections_topic': '/detections',
            'target_bbox_topic': '/target_bbox',
            'show_window': LaunchConfiguration('show_viewer_window'),
            'publish_debug_image': LaunchConfiguration('publish_debug_image'),
            'font_scale': 0.6,
            'line_thickness': 2,
        }],
        output='screen',
        emulate_tty=True,
    )


    rviz_config = os.path.join(
        '/home/jetson/ros_ws/src/ltm_poc/rviz',
        'ltm_poc_view.rviz'
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen',
        emulate_tty=True,
    )

    return LaunchDescription([
        image_path, publish_rate,
        yolo_model, yolo_conf, imgsz,
        target_text, clip_score_th, yolo_min_box, fov_deg,
        angular_kp, linear_speed, conf_thresh,
        show_viewer_window, publish_debug_image, start_image_view,

        img_pub,
        detector,
        vlm_clip,
        controller,
        turtlesim_node,
        teleport_up,
        viewer,
        rviz,
    ])
