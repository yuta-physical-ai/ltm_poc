
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import numpy as np
import torch
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from std_msgs.msg import String, Float32MultiArray
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose

import open_clip
from PIL import Image as PILImage

class VLMClipNode(Node):
    def __init__(self):
        super().__init__('vlm_clip_node')
        self.image_topic = self.declare_parameter('image_topic', '/camera/image_raw').value
        self.dets_topic  = self.declare_parameter('detections_topic', '/detections').value
        self.target_text = self.declare_parameter('target_text', 'red object').value

        self.model_name = self.declare_parameter('model_name', 'ViT-B-32').value
        self.pretrained = self.declare_parameter('pretrained', 'openai').value
        self.device_str = self.declare_parameter('device', '').value
        self.max_eval   = int(self.declare_parameter('max_eval', 16).value)
        self.min_box_conf = float(self.declare_parameter('min_box_conf', 0.10).value)
        self.score_threshold = float(self.declare_parameter('score_threshold', 0.25).value)
        self.fov_deg = float(self.declare_parameter('fov_deg', 60.0).value)
        self.publish_scores = bool(self.declare_parameter('publish_scores', False).value)

        self.device = torch.device(self.device_str) if self.device_str else torch.device('cpu')
        self.get_logger().info(f'Loading CLIP: {self.model_name}/{self.pretrained} on {self.device}')
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        self.model.eval()

        self.text_feat = None
        self._update_text(self.target_text)

        self.bridge = CvBridge()
        self.last_image = None

        self.sub_img  = self.create_subscription(Image, self.image_topic, self.on_image, 10)
        self.sub_dets = self.create_subscription(Detection2DArray, self.dets_topic, self.on_dets, 10)
        self.sub_goal = self.create_subscription(String, '/lang_goal', self.on_goal, 10)

        self.pub_scores  = self.create_publisher(Float32MultiArray, '/vlm_scores', 10)
        self.pub_ref     = self.create_publisher(String, '/vlm_ref_result', 10)
        self.pub_bbox    = self.create_publisher(Detection2D, '/target_bbox', 10)
        self.pub_dets_ex = self.create_publisher(Detection2DArray, '/detections_with_clip', 10)

    def on_image(self, msg: Image):
        try:
            self.last_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge: {e}')

    def on_goal(self, msg: String):
        t = msg.data.strip()
        if t and t != self.target_text:
            self.target_text = t
            self._update_text(self.target_text)
            self.get_logger().info(f'lang_goal -> "{self.target_text}"')

    @torch.no_grad()
    def on_dets(self, msg: Detection2DArray):
        if self.last_image is None or not msg.detections:
            self._publish_ref(0.0, 0.0)
            return

        dets = msg.detections[: self.max_eval]
        H, W = self.last_image.shape[:2]

        crops = []
        keep = []
        for d in dets:
            conf = 0.0
            if d.results:
                try:
                    conf = float(d.results[0].hypothesis.score)
                except Exception:
                    conf = 0.0
            if conf < self.min_box_conf or d.bbox is None:
                crops.append(None); keep.append(d); continue
            cx = float(d.bbox.center.position.x)
            cy = float(d.bbox.center.position.y)
            bw = float(d.bbox.size_x); bh = float(d.bbox.size_y)
            x1 = max(int(cx - bw/2), 0); y1 = max(int(cy - bh/2), 0)
            x2 = min(int(cx + bw/2), W); y2 = min(int(cy + bh/2), H)
            if x2<=x1 or y2<=y1:
                crops.append(None); keep.append(d); continue
            crop = self.last_image[y1:y2, x1:x2, ::-1]  # BGR->RGB
            crops.append(crop); keep.append(d)

        tens = [self.preprocess(PILImage.fromarray(c)).unsqueeze(0) if c is not None else None for c in crops]
        sims = []
        for t in tens:
            if t is None:
                sims.append(0.0); continue
            t = t.to(self.device)
            img_feat = self.model.encode_image(t)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            s = float((img_feat @ self.text_feat.T).squeeze().item())
            sims.append((s + 1.0) / 2.0)

        if self.publish_scores:
            m = Float32MultiArray(); m.data = sims; self.pub_scores.publish(m)

        dets_ex = []
        for d, s in zip(keep, sims):
            d2 = Detection2D()
            d2.bbox = d.bbox
            d2.results = list(d.results)
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = "clip"
            hyp.hypothesis.score = float(s)
            d2.results.append(hyp)
            dets_ex.append(d2)
        ex = Detection2DArray(); ex.header = msg.header; ex.detections = dets_ex
        self.pub_dets_ex.publish(ex)

        if not sims:
            self._publish_ref(0.0, 0.0); return
        best_idx = int(np.argmax(np.array(sims)))
        best_score = float(sims[best_idx])
        best_det = dets_ex[best_idx]

        if best_score < self.score_threshold:
            self._publish_ref(0.0, 0.0); return

        self.pub_bbox.publish(best_det)
        cx = float(best_det.bbox.center.position.x)
        ex_px = cx - (W/2.0)
        bearing = (ex_px / (W/2.0)) * (self.fov_deg/2.0)
        self._publish_ref(best_score, bearing)

    def _update_text(self, text: str):
        with torch.no_grad():
            toks = self.tokenizer([text]).to(self.device)
            feat = self.model.encode_text(toks)
            self.text_feat = feat / feat.norm(dim=-1, keepdim=True)

    def _publish_ref(self, confidence: float, bearing_deg: float):
        m = String()
        m.data = json.dumps({"confidence": float(confidence), "bearing_deg": float(bearing_deg)})
        self.pub_ref.publish(m)

def main():
    rclpy.init()
    n = VLMClipNode()
    try:
        rclpy.spin(n)
    finally:
        n.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
