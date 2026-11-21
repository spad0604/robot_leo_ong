import json
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import rclpy
import torch
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from ultralytics import YOLO


class CrackDetectorNode(Node):
    """Run one inference at a time and always consume the newest camera frame."""

    def __init__(self):
        super().__init__("crack_detector")

        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("model_path", "")
        self.declare_parameter("confidence_threshold", 0.35)
        self.declare_parameter("inference_size", 416)
        self.declare_parameter("preview_width", 640)
        self.declare_parameter("cpu_threads", 4)
        self.declare_parameter("device", "cpu")
        self.declare_parameter("output_dir", "~/crack_records")
        self.declare_parameter("save_interval_sec", 5.0)
        self.declare_parameter("crop_padding", 0.10)
        self.declare_parameter("log_every_n_frames", 5)

        model_path = self.get_parameter("model_path").value
        if not model_path:
            share_dir = Path(get_package_share_directory("crack_detector"))
            ncnn_model = share_dir / "weights_ncnn_model"
            model_path = str(ncnn_model if ncnn_model.is_dir() else share_dir / "weights.pt")
        model_path = Path(model_path).expanduser()
        if not model_path.exists():
            raise FileNotFoundError(f"YOLO model not found: {model_path}")

        self.confidence_threshold = float(self.get_parameter("confidence_threshold").value)
        self.inference_size = int(self.get_parameter("inference_size").value)
        self.preview_width = max(0, int(self.get_parameter("preview_width").value))
        self.save_interval_sec = max(0.0, float(self.get_parameter("save_interval_sec").value))
        self.crop_padding = max(0.0, float(self.get_parameter("crop_padding").value))
        self.log_every_n_frames = max(1, int(self.get_parameter("log_every_n_frames").value))
        self.output_dir = Path(self.get_parameter("output_dir").value).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        cpu_threads = max(1, int(self.get_parameter("cpu_threads").value))
        torch.set_num_threads(cpu_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # PyTorch only permits this before parallel work has started.
            pass
        cv2.setNumThreads(1)

        self.bridge = CvBridge()
        self.model = YOLO(str(model_path), task="detect")
        self.last_saved_at = None
        self.processed_frames = 0

        self.annotated_pub = self.create_publisher(Image, "/crack_detector/image_annotated", 1)
        self.detections_pub = self.create_publisher(String, "/crack_detector/detections", 10)
        image_topic = str(self.get_parameter("image_topic").value)
        self.image_sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
        )

        self._condition = threading.Condition()
        self._pending_msg = None
        self._pending_seq = 0
        self._dropped_frames = 0
        self._stop_worker = False
        self._worker = threading.Thread(
            target=self._inference_worker,
            name="crack-inference",
            daemon=True,
        )
        self._worker.start()

        self.get_logger().info(f"Model: {model_path}")
        self.get_logger().info(f"Subscribing to: {image_topic}")
        self.get_logger().info(
            f"Latest-frame mode; inference_size={self.inference_size}, CPU threads={cpu_threads}"
        )
        self.get_logger().info(f"Saving crack crops under: {self.output_dir}")

    def image_callback(self, msg):
        # Replace an unprocessed image instead of queueing stale frames.
        with self._condition:
            if self._pending_msg is not None:
                self._dropped_frames += 1
            self._pending_msg = msg
            self._pending_seq += 1
            self._condition.notify()

    def _inference_worker(self):
        while True:
            with self._condition:
                while self._pending_msg is None and not self._stop_worker:
                    self._condition.wait()
                if self._stop_worker:
                    return
                msg = self._pending_msg
                self._pending_msg = None
                dropped_frames = self._dropped_frames
                self._dropped_frames = 0

            try:
                self._process_image(msg, dropped_frames)
            except Exception as exc:
                self.get_logger().error(f"Image inference failed: {exc}")

    def _process_image(self, msg, dropped_frames):
        started_at = time.monotonic()
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        result = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            imgsz=self.inference_size,
            device=str(self.get_parameter("device").value),
            verbose=False,
        )[0]

        detections = []
        boxes = result.boxes
        if boxes is not None:
            height, width = frame.shape[:2]
            for box in boxes:
                x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
                detections.append(
                    {
                        "class": result.names[int(box.cls[0].item())],
                        "confidence": float(box.conf[0].item()),
                        "bbox_xyxy": [x1, y1, x2, y2],
                        "_crop": self._crop_with_padding(frame, x1, y1, x2, y2, width, height),
                    }
                )

        now = time.monotonic()
        should_save = bool(detections) and (
            self.last_saved_at is None or now - self.last_saved_at >= self.save_interval_sec
        )
        for index, detection in enumerate(detections):
            crop = detection.pop("_crop")
            if should_save:
                saved_path = self._save_crop(crop, detection["confidence"], index)
                detection["saved_crop"] = str(saved_path) if saved_path else ""
        if should_save:
            self.last_saved_at = now

        annotated_frame = result.plot()
        if self.preview_width and annotated_frame.shape[1] > self.preview_width:
            preview_height = round(
                annotated_frame.shape[0] * self.preview_width / annotated_frame.shape[1]
            )
            annotated_frame = cv2.resize(
                annotated_frame,
                (self.preview_width, preview_height),
                interpolation=cv2.INTER_AREA,
            )
        annotated = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
        annotated.header = msg.header
        self.annotated_pub.publish(annotated)

        self.processed_frames += 1
        processing_ms = (time.monotonic() - started_at) * 1000.0
        frame_age_ms = self._frame_age_ms(msg)
        payload = {
            "stamp": {"sec": msg.header.stamp.sec, "nanosec": msg.header.stamp.nanosec},
            "frame_id": msg.header.frame_id,
            "timing": {
                "processing_ms": round(processing_ms, 1),
                "frame_age_ms": round(frame_age_ms, 1),
                "dropped_frames": dropped_frames,
            },
            "detections": detections,
        }
        self.detections_pub.publish(String(data=json.dumps(payload)))

        if self.processed_frames % self.log_every_n_frames == 0:
            self.get_logger().info(
                f"frame={self.processed_frames} inference={processing_ms:.0f} ms "
                f"age={frame_age_ms:.0f} ms dropped={dropped_frames} "
                f"detections={len(detections)}"
            )

    def _frame_age_ms(self, msg):
        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        now_ns = self.get_clock().now().nanoseconds
        return max(0.0, (now_ns - stamp_ns) / 1_000_000.0)

    def _crop_with_padding(self, frame, x1, y1, x2, y2, width, height):
        box_width = max(1, x2 - x1)
        box_height = max(1, y2 - y1)
        pad_x = int(box_width * self.crop_padding)
        pad_y = int(box_height * self.crop_padding)
        left = max(0, x1 - pad_x)
        top = max(0, y1 - pad_y)
        right = min(width, x2 + pad_x)
        bottom = min(height, y2 + pad_y)
        return frame[top:bottom, left:right]

    def _save_crop(self, crop, confidence, index):
        if crop is None or crop.size == 0:
            return None

        now = datetime.now()
        date_dir = self.output_dir / now.strftime("%Y-%m-%d")
        date_dir.mkdir(parents=True, exist_ok=True)
        timestamp = now.strftime("%Y%m%d_%H%M%S_%f")
        path = date_dir / f"crack_{timestamp}_{index:02d}_{confidence:.2f}.jpg"
        if not cv2.imwrite(str(path), crop):
            self.get_logger().error(f"Could not write crop: {path}")
            return None
        self.get_logger().info(f"Saved crack crop: {path}")
        return path

    def destroy_node(self):
        if hasattr(self, "_condition"):
            with self._condition:
                self._stop_worker = True
                self._condition.notify_all()
            if hasattr(self, "_worker"):
                self._worker.join(timeout=30.0)
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CrackDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
