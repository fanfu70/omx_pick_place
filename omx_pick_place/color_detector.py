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
        self.annotated_image_pub = self.create_publisher(Image, '/color_detector/annotated_image', 10)
        
        # Subscriber
        self.image_sub = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.image_callback, 10
        )

        self.get_logger().info(f"Color detector started. Looking for: {self.target_color}")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            return

        # Make a copy for annotation
        annotated_image = cv_image.copy()

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
            # Still publish annotated image (without detection)
            self._publish_annotated_image(annotated_image, msg.header)
            return

        u, v = centroid
        # Adjust for crosshair offset (if needed)
        u -= 15
        v -= 20

        # Draw detection on annotated image
        self._draw_detection(annotated_image, u, v)

        # Rate limiting for centroid/marker publishing
        current_time = self.get_clock().now()
        time_since_last = (current_time - self.last_pub_time).nanoseconds / 1e9
        if time_since_last < self.min_pub_interval:
            # Still publish annotated image every frame
            self._publish_annotated_image(annotated_image, msg.header)
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

        # Publish annotated image
        self._publish_annotated_image(annotated_image, msg.header)

        self.get_logger().debug(f"Detected object at pixel ({u}, {v})")

    def _draw_detection(self, image, u, v):
        """Draw detection results on the image."""
        u, v = int(u), int(v)
        
        # Draw crosshair
        crosshair_length = 30
        crosshair_thickness = 2
        color = (0, 255, 0)  # Green in BGR
        
        # Horizontal line
        cv2.line(image, (u - crosshair_length, v), (u + crosshair_length, v), color, crosshair_thickness)
        # Vertical line
        cv2.line(image, (u, v - crosshair_length), (u, v + crosshair_length), color, crosshair_thickness)
        
        # Draw circle around detection
        circle_radius = 20
        cv2.circle(image, (u, v), circle_radius, color, 2)
        
        # Draw filled center dot
        cv2.circle(image, (u, v), 5, (0, 0, 255), -1)  # Red dot in BGR
        
        # Draw text with coordinates
        text = f"({u}, {v})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
        
        # Draw background rectangle
        rect_x = u + 10
        rect_y = v - 30
        cv2.rectangle(image, 
                     (rect_x - 5, rect_y - text_height - 5), 
                     (rect_x + text_width + 5, rect_y + baseline + 5), 
                     (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(image, text, (rect_x, rect_y), font, font_scale, color, font_thickness)

    def _publish_annotated_image(self, cv_image, header):
        """Publish the annotated camera image."""
        try:
            image_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            image_msg.header = header
            self.annotated_image_pub.publish(image_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish annotated image: {e}")

    def _create_2d_marker(self, u: int, v: int, frame_id: str) -> Marker:
        """Create a 2D marker for visualization in RViz."""
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "detection"
        marker.id = 0
        marker.type = Marker.SPHERE
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
