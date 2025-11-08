
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String

class ControllerNode(Node):
    def __init__(self):
        super().__init__('controller_node')
        self.angular_kp   = float(self.declare_parameter('angular_kp', 0.8).value)
        self.linear_speed = float(self.declare_parameter('linear_speed', 0.35).value)
        self.conf_thresh  = float(self.declare_parameter('conf_thresh', 0.05).value)

        self.pub = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)
        self.sub = self.create_subscription(String, '/vlm_ref_result', self.on_ref, 10)

    def on_ref(self, msg: String):
        try:
            d = json.loads(msg.data)
        except Exception:
            return
        conf = float(d.get('confidence', 0.0))
        bearing_deg = float(d.get('bearing_deg', 0.0))

        t = Twist()
        if conf >= self.conf_thresh:
            rad = math.radians(bearing_deg)
            t.angular.z = -1 * max(-1.5, min(1.5, self.angular_kp * rad))
            t.linear.x = self.linear_speed
        self.pub.publish(t)

def main():
    rclpy.init()
    n = ControllerNode()
    try:
        rclpy.spin(n)
    finally:
        n.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
