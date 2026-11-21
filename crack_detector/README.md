# ROS 2 crack detector

ROS 2 Jazzy package for a Raspberry Pi CSI camera and the supplied YOLO11n `weights.pt`. The model has one detection class: `crack`. The launch file uses the matching NCNN CPU export (`weights_ncnn_model`) at 416px for lower latency; the original PyTorch model is retained as the export source and fallback.

The launch file starts two nodes:

- `rpicam_camera` reads MJPEG from `rpicam-vid` and publishes `/camera/image_raw`.
- `crack_detector` publishes annotated frames on `/crack_detector/image_annotated`, JSON detections on `/crack_detector/detections`, and saves padded crack crops under `~/crack_records/YYYY-MM-DD/`.

The detector saves at most one set of crops every five seconds while cracks remain visible. Adjust `save_interval_sec` to change this interval; set it to `0.0` to save crops from every detected frame.

## Build and run on the Raspberry Pi

Copy this package into a ROS 2 workspace, then install the Python runtime dependencies for the same Python interpreter used by ROS. On Raspberry Pi OS Trixie, Debian provides ARM64 `python3-torch` and `python3-torchvision`; use those system packages with the ROS Python packages to avoid replacing `rclpy`/`cv_bridge` dependencies with a separate Python environment.

```bash
sudo apt install ros-jazzy-cv-bridge colcon python3-colcon-ros \
  python3-colcon-package-selection python3-colcon-package-information \
  python3-colcon-python-setup-py \
  python3-opencv python3-numpy python3-torch python3-torchvision \
  python3-pil python3-yaml python3-matplotlib python3-requests \
  python3-scipy python3-psutil
```

Install only the Ultralytics Python package. `--no-deps` is important here: it keeps pip from replacing Debian's CPU PyTorch with large CUDA/NVIDIA wheels. Debian Trixie provides ARM64 `python3-torch` and `python3-torchvision` packages.

```bash
python3 -m pip install --break-system-packages --no-deps --no-cache-dir ultralytics==8.4.55 ultralytics-thop
python3 -m pip install --break-system-packages --no-deps --no-cache-dir ncnn==1.0.20260526 pnnx==20260526
python3 -m pip install --break-system-packages --no-cache-dir polars
```

Copy this package directory to `~/ros2_ws/src/crack_detector`, then build and launch:

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
ros2 launch crack_detector crack_detection.launch.py
```

The package includes both `weights.pt` and its NCNN export; no separate model download is needed. `rpicam-vid` must be installed and able to use the CSI camera. Run `rpicam-hello --list-cameras` first if the camera node cannot start.

The camera publisher and detector both keep the stream current: `rpicam-vid` flushes each frame, buffered JPEGs and unprocessed ROS images are dropped when a stage falls behind, and the annotated preview is reduced to 640px wide to cut viewer/network load. A slower inference rate therefore skips frames instead of building a display delay. `inference_size`, `preview_width`, and `confidence_threshold` can be changed in the launch file. Preview annotated detections from another terminal with:

```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 run image_tools showimage --ros-args --remap image:=/crack_detector/image_annotated
```

## Useful parameters

Run the detector with an already-published image topic:

```bash
ros2 run crack_detector crack_detector --ros-args \
  -p image_topic:=/camera/image_raw \
  -p model_path:=/path/to/weights_ncnn_model \
  -p inference_size:=416 \
  -p confidence_threshold:=0.60 \
  -p save_interval_sec:=5.0 \
  -p output_dir:=/home/robot/crack_records
```

The camera publisher can also run on its own:

```bash
ros2 run crack_detector rpicam_camera
```
