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
        self.declare_parameter('min_contour_area', 100)
        self.declare_parameter('pub_rate_hz', 10.0)  # Publish at most N times per second
        self.target_color = self.get_parameter('target_color').value
        self.min_contour_area = self.get_parameter('min_contour_area').value
        self.pub_rate_hz = self.get_parameter('pub_rate_hz').value

        # Rate limiting
        self.last_pub_time = self.get_clock().now()
        self.min_pub_interval = 1.0 / self.pub_rate_hz if self.pub_rate_hz > 0 else 0.0

        # Publishers
        self.centroid_pub = self.create_publisher(Pose2D, '/detected_object', 10)
        self.marker_pub = self.create_publisher(Marker, '/detection_marker', 10)
        
        # Subscriber
        self.image_sub = self.create_subscription(
            Image, '/realsense/color/image_raw', self.image_callback, 10
        )

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
        centroid = find_largest_contour_centroid(cleaned_mask, min_area=self.min_contour_area)

        if centroid is None:
            # Remove old marker if no detection
            self._publish_empty_marker(msg.header.frame_id)
            return

        u, v = centroid

        # Rate limiting
        current_time = self.get_clock().now()
        time_since_last = (current_time - self.last_pub_time).nanoseconds / 1e9
        if time_since_last < self.min_pub_interval:
            return

        self.last_pub_time = current_time

        # Publish Centroid
        centroid_msg = Pose2D()
        centroid_msg.x = float(u)
        centroid_msg.y = float(v)
        self.centroid_pub.publish(centroid_msg)

        # Publish RViz Marker (2D circle in image coordinates for debugging)
        marker = self._create_2d_marker(u, v, msg.header.frame_id)
        self.marker_pub.publish(marker)

        self.get_logger().debug(f"Detected object at pixel ({u}, {v})")

    def _create_2d_marker(self, u: int, v: int, frame_id: str) -> Marker:
        """Create a 2D marker for visualization in RViz."""
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "detection"
        marker.id = 0
        marker.type = Marker.CIRCLE
        marker.action = Marker.ADD
        marker.pose.position.x = float(u)
        marker.pose.position.y = float(v)
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 20.0  # radius in pixels
        marker.scale.y = 1.0   # line width
        marker.scale.z = 0.0
        marker.color.a = 0.8
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        return marker

    def _publish_empty_marker(self, frame_id: str):
        """Publish an empty marker to clear previous visualization."""
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "detection"
        marker.id = 0
        marker.action = Marker.DELETE
        self.marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = ColorDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Clear marker on shutdown
        marker = Marker()
        marker.ns = "detection"
        marker.id = 0
        marker.action = Marker.DELETE
        node.marker_pub.publish(marker)
        node.destroy_node()
        rclpy.shutdown()
