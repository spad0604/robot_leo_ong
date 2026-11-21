from glob import glob
from setuptools import find_packages, setup

package_name = "crack_detector"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "weights.pt"]),
        ("share/" + package_name + "/weights_ncnn_model", glob("weights_ncnn_model/*.*")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="robot",
    maintainer_email="robot@example.com",
    description="ROS 2 crack detection with a Raspberry Pi CSI camera and YOLO.",
    license="Proprietary",
    entry_points={
        "console_scripts": [
            "crack_detector = crack_detector.detector_node:main",
            "rpicam_camera = crack_detector.rpicam_camera_node:main",
        ],
    },
)
