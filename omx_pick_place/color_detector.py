#!/usr/bin/env python3
"""
Color Detector Node:
Subscribes to RGB camera, finds colored object, publishes centroid (u, v).
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose2D
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge
import cv2
import numpy as np

from omx_pick_place.utils import get_color_mask, morphological_clean, find_largest_contour_centroid

class ColorDetector(Node):
    def __init__(self):
        super().__init__('color_detector')
        self.bridge = CvBridge()
        
        # Configuration
        self.declare_parameter('target_color', 'red')
        self.target_color = self.get_parameter('target_color').value
        
        # Publishers
        self.centroid_pub = self.create_publisher(Pose2D, '/detected_object', 10)
        self.marker_pub = self.create_publisher(Marker, '/detection_marker', 10)
        
        # Subscriber
        self.image_sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.image_callback, 10)
            
        self.get_logger().info(f"Color detector started. Looking for: {self.target_color}")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            return

        # 1. Convert to HSV
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        # 2. Get Mask
        mask = get_color_mask(hsv, self.target_color)
        
        # 3. Clean Mask
        cleaned_mask = morphological_clean(mask)
        
        # 4. Find Centroid
        centroid = find_largest_contour_centroid(cleaned_mask)
        
        if centroid:
            u, v = centroid
            
            # Publish Centroid
            centroid_msg = Pose2D()
            centroid_msg.x = float(u)
            centroid_msg.y = float(v)
            self.centroid_pub.publish(centroid_msg)
            
            # Publish RViz Marker (Circle at centroid)
            marker = Marker()
            marker.header.frame_id = msg.header.frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "detection"
            marker.id = 0
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = 0.0
            marker.pose.position.y = 0.0
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.scale.z = 0.05
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            self.marker_pub.publish(marker)
            
            # Optional: Draw on image and save for debugging (uncomment to use)
            # cv2.circle(cv_image, centroid, 10, (0, 255, 0), -1)
            # cv2.imshow("Detection", cv_image)
            # cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = ColorDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
