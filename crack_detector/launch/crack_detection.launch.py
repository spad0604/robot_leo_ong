from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    share_dir = Path(get_package_share_directory("crack_detector"))
    return LaunchDescription(
        [
            Node(
                package="crack_detector",
                executable="rpicam_camera",
                name="rpicam_camera",
                output="screen",
                parameters=[
                    {
                        "image_topic": "/camera/image_raw",
                        "width": 1280,
                        "height": 720,
                        "framerate": 5,
                    }
                ],
            ),
            Node(
                package="crack_detector",
                executable="crack_detector",
                name="crack_detector",
                output="screen",
                parameters=[
                    {
                        "image_topic": "/camera/image_raw",
                        "model_path": str(share_dir / "weights_ncnn_model"),
                        "inference_size": 416,
                        "preview_width": 640,
                        "confidence_threshold": 0.60,
                        "cpu_threads": 4,
                        "output_dir": "~/crack_records",
                    }
                ],
            ),
        ]
    )
