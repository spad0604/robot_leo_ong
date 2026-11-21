import os
import select
import shutil
import subprocess
import threading

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


class RPiCamCameraNode(Node):
    """Publish MJPEG frames from rpicam-vid as sensor_msgs/Image."""

    def __init__(self):
        super().__init__("rpicam_camera")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("width", 1280)
        self.declare_parameter("height", 720)
        self.declare_parameter("framerate", 5)
        self.declare_parameter("frame_id", "camera_optical_frame")

        executable = shutil.which("rpicam-vid")
        if executable is None:
            raise RuntimeError("rpicam-vid not found; install Raspberry Pi camera applications")

        width = int(self.get_parameter("width").value)
        height = int(self.get_parameter("height").value)
        framerate = int(self.get_parameter("framerate").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        image_topic = str(self.get_parameter("image_topic").value)
        self.publisher = self.create_publisher(
            Image,
            image_topic,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
        )
        self.bridge = CvBridge()
        self.stop_event = threading.Event()
        self._published_frames = 0
        self._skipped_jpegs = 0

        command = [
            executable,
            "--timeout",
            "0",
            "--nopreview",
            "--codec",
            "mjpeg",
            "--width",
            str(width),
            "--height",
            str(height),
            "--framerate",
            str(framerate),
            "--flush",
            "--output",
            "-",
        ]
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        os.set_blocking(self.process.stdout.fileno(), False)
        self.reader_thread = threading.Thread(target=self._read_frames, daemon=True)
        self.reader_thread.start()

        self.get_logger().info(f"Publishing camera images on {image_topic}")
        self.get_logger().info(f"rpicam-vid: {width}x{height} at {framerate} fps")

    def _read_frames(self):
        buffer = bytearray()
        fd = self.process.stdout.fileno()
        while not self.stop_event.is_set():
            readable, _, _ = select.select([fd], [], [], 0.5)
            if not readable:
                if self.process.poll() is not None:
                    break
                continue

            stream_ended = False
            while not self.stop_event.is_set():
                try:
                    chunk = os.read(fd, 262144)
                except BlockingIOError:
                    break
                if not chunk:
                    stream_ended = True
                    break
                buffer.extend(chunk)

            latest_jpeg = None
            latest_received_at = None

            while not self.stop_event.is_set():
                start = buffer.find(b"\xff\xd8")
                if start < 0:
                    if len(buffer) > 1:
                        del buffer[:-1]
                    break
                if start > 0:
                    del buffer[:start]
                end = buffer.find(b"\xff\xd9", 2)
                if end < 0:
                    break
                if latest_jpeg is not None:
                    self._skipped_jpegs += 1
                latest_jpeg = bytes(buffer[: end + 2])
                latest_received_at = self.get_clock().now().to_msg()
                del buffer[: end + 2]

            # If rpicam-vid or inference falls behind, discard complete JPEGs
            # already waiting in the pipe and publish only the newest one.
            if latest_jpeg is not None:
                self._publish_jpeg(latest_jpeg, latest_received_at)
            if stream_ended:
                break

            if len(buffer) > 8 * 1024 * 1024:
                buffer.clear()

        if not self.stop_event.is_set():
            self.get_logger().error("rpicam-vid stopped; check camera connection and permissions")

    def _publish_jpeg(self, jpeg, received_at):
        encoded = np.frombuffer(jpeg, dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None:
            return
        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = received_at
        msg.header.frame_id = self.frame_id
        self.publisher.publish(msg)
        self._published_frames += 1
        if self._published_frames % 25 == 0:
            self.get_logger().info(
                f"Camera frames={self._published_frames}, skipped buffered JPEGs={self._skipped_jpegs}"
            )

    def destroy_node(self):
        self.stop_event.set()
        if hasattr(self, "process") and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2.0)
        if hasattr(self, "reader_thread"):
            self.reader_thread.join(timeout=2.0)
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RPiCamCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
