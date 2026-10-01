#!/usr/bin/env python3

import sys
import cv2
import torch
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge


# ---------------------------------------------------------
# YOLOv5 paths
# ---------------------------------------------------------

YOLOV5_DIR = "/home/devilfisher/.cache/torch/hub/ultralytics_yolov5_master"

# CHANGE THIS TO YOUR MODEL
WEIGHTS = "/home/devilfisher/Downloads/custom healthy or injured segmentation model.pt"

sys.path.insert(0, YOLOV5_DIR)

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import (
    non_max_suppression,
    scale_boxes,
    check_img_size
)
from utils.segment.general import process_mask_native
from utils.torch_utils import select_device


class YOLOv5Segmenter(Node):

    def __init__(self):
        super().__init__('yolov5_segmenter')

        # -------------------------------------------------
        # ROS camera
        # -------------------------------------------------

        self.camera_topic = (
            '/world/map/model/iris/link/'
            'pitch_link/sensor/camera/image'
        )

        self.subscription = self.create_subscription(
            Image,
            self.camera_topic,
            self.image_callback,
            qos_profile_sensor_data
        )

        self.publisher_ = self.create_publisher(
            String,
            '/detected_objects',
            10
        )

        self.bridge = CvBridge()

        # -------------------------------------------------
        # YOLO segmentation model
        # -------------------------------------------------

        self.get_logger().info(
            "Loading YOLOv5 segmentation model..."
        )

        self.device = select_device('')

        self.model = DetectMultiBackend(
            WEIGHTS,
            device=self.device,
            dnn=False,
            fp16=False
        )

        self.stride = self.model.stride
        self.names = self.model.names

        self.imgsz = check_img_size(
            (640, 640),
            s=self.stride
        )

        self.conf_threshold = 0.10
        self.iou_threshold = 0.10

        # Warmup
        self.model.warmup(
            imgsz=(1, 3, self.imgsz[0], self.imgsz[1])
        )

        self.get_logger().info(
            f"Segmentation model loaded. Classes: {self.names}"
        )

        self.get_logger().info(
            f"Listening to camera: {self.camera_topic}"
        )

    # =====================================================
    # CAMERA CALLBACK
    # =====================================================

    def image_callback(self, msg):

        try:

            # ROS Image -> OpenCV BGR
            camera_image = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8'
            )

        except Exception as e:

            self.get_logger().error(
                f"CvBridge error: {e}"
            )

            return

        
        # Image where segmentation results will be drawn
        result_image = camera_image.copy()

        # -------------------------------------------------
        # YOLO PREPROCESSING
        # -------------------------------------------------

        # Resize + letterbox while preserving aspect ratio
        img = letterbox(
            camera_image,
            self.imgsz,
            stride=self.stride,
            auto=True
        )[0]

        # BGR -> RGB
        # HWC -> CHW
        img = img.transpose((2, 0, 1))[::-1]

        img = np.ascontiguousarray(img)

        # numpy -> torch
        img = torch.from_numpy(img).to(self.device)

        img = img.float()

        # 0-255 -> 0.0-1.0
        img /= 255.0

        # CHW -> BCHW
        if img.ndim == 3:
            img = img.unsqueeze(0)

        # -------------------------------------------------
        # YOLO SEGMENTATION INFERENCE
        # -------------------------------------------------

        with torch.no_grad():

            output = self.model(img)

            # Segmentation model gives:
            #
            # predictions
            # prototype masks
            pred, proto = output[:2]

        # -------------------------------------------------
        # NON-MAX SUPPRESSION
        # -------------------------------------------------

        pred = non_max_suppression(
            pred,
            self.conf_threshold,
            self.iou_threshold,
            max_det=100,
            nm=32
        )

        detected_objects = []

        # Only one camera image, so batch index = 0
        det = pred[0]

        # -------------------------------------------------
        # PROCESS DETECTIONS
        # -------------------------------------------------

        if len(det):

            # Scale bounding boxes back to original camera
            # resolution FIRST
            det[:, :4] = scale_boxes(
                img.shape[2:],
                det[:, :4],
                camera_image.shape
            ).round()

            # Generate masks at original image resolution
            masks = process_mask_native(
                proto[0],
                det[:, 6:],
                det[:, :4],
                camera_image.shape[:2]
            )

            # -------------------------------------------------
            # DRAW EVERY DETECTION
            # -------------------------------------------------

            for i, detection in enumerate(det[:, :6]):

                x1, y1, x2, y2, conf, cls = detection

                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)

                confidence = float(conf)
                class_id = int(cls)

                class_name = self.names[class_id]

                # -----------------------------------------
                # Draw segmentation mask
                # -----------------------------------------

                mask = masks[i].cpu().numpy().astype(bool)

                overlay = result_image.copy()

                overlay[mask] = (0, 255, 0)

                result_image = cv2.addWeighted(
                    result_image,
                    0.7,
                    overlay,
                    0.3,
                    0
                )

                # -----------------------------------------
                # Draw bounding box
                # -----------------------------------------

                cv2.rectangle(
                    result_image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                label = (
                    f"{class_name} "
                    f"{confidence:.2f}"
                )

                cv2.putText(
                    result_image,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                self.get_logger().info(
                    f"Detected: "
                    f"{class_name} "
                    f"confidence={confidence:.2f}"
                )

                detected_objects.append(
                    f"{class_name} {confidence:.2f}"
                )

        # -------------------------------------------------
        # ROS detection publisher
        # -------------------------------------------------

        output_msg = String()

        if detected_objects:

            output_msg.data = ", ".join(
                detected_objects
            )

        else:

            output_msg.data = (
                "No objects detected"
            )

        self.publisher_.publish(output_msg)

        # -------------------------------------------------
        # SHOW SEGMENTATION RESULT
        # -------------------------------------------------

        cv2.imshow(
            "YOLOv5 Segmentation",
            result_image
        )

        cv2.waitKey(1)


def main(args=None):

    rclpy.init(args=args)

    node = YOLOv5Segmenter()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        rclpy.shutdown()

        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
