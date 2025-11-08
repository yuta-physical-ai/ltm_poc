#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
from typing import Optional, Tuple

import cv2
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D

def _get_cls_score(r):
    h = getattr(r, 'hypothesis', None)
    if h:
        cid = getattr(h, 'class_id', '')
        sc  = getattr(h, 'score', None)
    else:
        cid = getattr(r, 'class_id', '')
        sc  = getattr(r, 'score', None)
    return str(cid), float(sc) if sc is not None else None

class ViewerNode(Node):
    def __init__(self):
        super().__init__('viewer_node')
        self.image_topic  = self.declare_parameter('image_topic', '/camera/image_raw').value
        self.dets_topic   = self.declare_parameter('detections_topic', '/detections_with_clip').value
        self.target_topic = self.declare_parameter('target_bbox_topic', '/target_bbox').value

        self.show_window   = bool(self.declare_parameter('show_window', False).value)
        self.publish_debug = bool(self.declare_parameter('publish_debug_image', True).value)
        self.font_scale    = float(self.declare_parameter('font_scale', 0.6).value)
        self.thickness     = int(self.declare_parameter('line_thickness', 2).value)

        self.bridge = CvBridge()
        self.last_img: Optional[Image] = None
        self.last_dets: Optional[Detection2DArray] = None
        self.last_target: Optional[Detection2D] = None
        self._last_t = time.time()
        self._cnt = 0
        self._fps = 0.0

        self.sub_img  = self.create_subscription(Image, self.image_topic, self.on_image, 10)
        self.sub_dets = self.create_subscription(Detection2DArray, self.dets_topic, self.on_dets, 10)
        self.sub_tgt  = self.create_subscription(Detection2D, self.target_topic, self.on_target, 10)
        self.pub_debug = self.create_publisher(Image, '/debug/overlay', 10)

        if self.show_window:
            try:
                cv2.namedWindow('viewer', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('viewer', 960, 540)
            except Exception as e:
                self.get_logger().warn(f'cv2 window init failed: {e}')
                self.show_window = False

        self.get_logger().info(
            f'viewer_node ready: img={self.image_topic}, dets={self.dets_topic}, '
            f'target={self.target_topic}, show_window={self.show_window}'
        )

    def on_image(self, msg: Image):
        try:
            self.last_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge: {e}')
            return
        self._render()

    def on_dets(self, msg: Detection2DArray):
        self.last_dets = msg

    def on_target(self, msg: Detection2D):
        self.last_target = msg

    def _render(self):
        if self.last_img is None:
            return
        img = self.last_img.copy()
        H, W = img.shape[:2]

        self._cnt += 1
        now = time.time()
        if now - self._last_t >= 1.0:
            self._fps = self._cnt / (now - self._last_t)
            self._cnt = 0
            self._last_t = now

        if self.last_dets and self.last_dets.detections:
            for d in self.last_dets.detections:
                if not d.bbox:
                    continue
                cx = float(d.bbox.center.position.x)
                cy = float(d.bbox.center.position.y)
                w  = float(d.bbox.size_x); h = float(d.bbox.size_y)
                x1 = max(int(cx - w/2), 0); y1 = max(int(cy - h/2), 0)
                x2 = min(int(cx + w/2), W-1); y2 = min(int(cy + h/2), H-1)
                cv2.rectangle(img, (x1, y1), (x2, y2), (60,200,60), self.thickness)

                yolo_txt = None
                if hasattr(d, 'results') and len(d.results) > 0:
                    try:
                        cls0, sc0 = _get_cls_score(d.results[0])
                        if sc0 is not None:
                            yolo_txt = f'YOLO: {cls0 if cls0 else "obj"} {sc0:.2f}'
                    except Exception:
                        pass

                if yolo_txt:
                    (tw, th), _ = cv2.getTextSize(yolo_txt, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.thickness)
                    y_bg_top = max(y1 - th - 6, 0)
                    cv2.rectangle(img, (x1, y_bg_top), (x1 + tw + 6, y1), (60,200,60), -1)
                    cv2.putText(img, yolo_txt, (x1 + 3, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (0,0,0), 1, lineType=cv2.LINE_AA)

        if self.last_target and self.last_target.bbox:
            d = self.last_target
            cx = float(d.bbox.center.position.x)
            cy = float(d.bbox.center.position.y)
            w  = float(d.bbox.size_x); h = float(d.bbox.size_y)
            x1 = max(int(cx - w/2), 0); y1 = max(int(cy - h/2), 0)
            x2 = min(int(cx + w/2), W-1); y2 = min(int(cy + h/2), H-1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), self.thickness+1)
            label_y = min(H - 5, y2 + 22)
            cv2.putText(img, 'TARGET', (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                        self.font_scale, (0,0,255), self.thickness)

        hud = f'{W}x{H} | {self._fps:.1f} FPS'
        cv2.putText(img, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        if self.show_window:
            try:
                cv2.imshow('viewer', img)
                cv2.waitKey(1)
            except Exception as e:
                self.get_logger().warn(f'cv2 imshow failed: {e}')

        if self.publish_debug:
            self.pub_debug.publish(self.bridge.cv2_to_imgmsg(img, encoding='bgr8'))

def main():
    rclpy.init()
    n = ViewerNode()
    try:
        rclpy.spin(n)
    finally:
        n.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
