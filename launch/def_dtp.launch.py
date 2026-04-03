"""ROS2 launch file for the DEF-DTP node."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("predictor", default_value="replay"),
            DeclareLaunchArgument("dataset", default_value="nuscenes"),
            DeclareLaunchArgument("device", default_value="auto"),
            Node(
                package="anima_def_dtp",
                executable="def_dtp_node",
                name="def_dtp",
                parameters=[
                    {
                        "predictor": LaunchConfiguration("predictor"),
                        "dataset": LaunchConfiguration("dataset"),
                        "device": LaunchConfiguration("device"),
                    }
                ],
                output="screen",
            ),
        ]
    )
