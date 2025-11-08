
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rclpy
from rclpy.node import Node
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose

from ultralytics import YOLO

class SimpleDetector(Node):
    def __init__(self):
        super().__init__('simple_detector_node')
        self.image_topic = self.declare_parameter('input_topic', '/camera/image_raw').value
        self.model_path  = self.declare_parameter('model', 'yolov8n.pt').value
        self.conf        = float(self.declare_parameter('conf', 0.25).value)
        self.imgsz       = int(self.declare_parameter('imgsz', 640).value)
        
        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, self.image_topic, self.on_image, 10)
        self.pub_dets = self.create_publisher(Detection2DArray, '/detections', 10)
        self.pub_img  = self.create_publisher(Image, '/yolo_image', 10)

        self.get_logger().info(f'Loading YOLO: {self.model_path}')
        self.model = YOLO(self.model_path)
        self.get_logger().info(f'ready: img={self.image_topic}, conf={self.conf}, imgsz={self.imgsz}')

    def on_image(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge: {e}')
            return

        try:
            results = self.model.predict(source=frame, imgsz=self.imgsz, conf=self.conf, verbose=False)
        except Exception as e:
            self.get_logger().error(f'YOLO: {e}')
            return
        if not results:
            return
        r0 = results[0]

        # draw preview
        im_draw = r0.plot()
        self.pub_img.publish(self.bridge.cv2_to_imgmsg(im_draw, encoding='bgr8'))

        out = Detection2DArray()
        out.header = msg.header
        if getattr(r0, "boxes", None):
            for b in r0.boxes:
                det = Detection2D()
                xywh = b.xywh[0].cpu().numpy().tolist()
                det.bbox.center.position.x = float(xywh[0])
                det.bbox.center.position.y = float(xywh[1])
                det.bbox.size_x = float(xywh[2])
                det.bbox.size_y = float(xywh[3])

                hyp = ObjectHypothesisWithPose()
                try:
                    hyp.hypothesis.class_id = str(int(b.cls.item()))
                except Exception:
                    hyp.hypothesis.class_id = str(b.cls.item())
                hyp.hypothesis.score = float(b.conf.item())

                det.results.append(hyp)
                out.detections.append(det)
        self.pub_dets.publish(out)

def main():
    rclpy.init()
    n = SimpleDetector()
    try:
        rclpy.spin(n)
    finally:
        n.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
