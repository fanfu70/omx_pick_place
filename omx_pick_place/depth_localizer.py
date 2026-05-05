#!/usr/bin/env python3
"""
Depth Localizer Node:
Takes 2D centroid and Depth image, computes 3D position, transforms to 'world' frame.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose2D, PoseStamped
from cv_bridge import CvBridge
import numpy as np
import tf2_ros
from tf2_geometry_msgs import do_transform_pose_stamped

class DepthLocalizer(Node):
    def __init__(self):
        super().__init__('depth_localizer')
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Publishers
        self.pose_pub = self.create_publisher(PoseStamped, '/object_pose', 10)
        
        # Subscribers
        self.centroid_sub = self.create_subscription(
            Pose2D, '/detected_object', self.centroid_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_rect_raw', self.depth_callback, 10)
        self.info_sub = self.create_subscription(
            CameraInfo, '/camera/color/camera_info', self.info_callback, 10)
            
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.latest_depth = None
        
        self.get_logger().info("Depth localizer started.")

    def info_callback(self, msg):
        # Camera intrinsics from K matrix
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]
        self.get_logger().info(f"Camera intrinsics loaded: fx={self.fx}, fy={self.fy}")

    def depth_callback(self, msg):
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.depth_frame_id = msg.header.frame_id

    def centroid_callback(self, msg):
        if self.fx is None or self.latest_depth is None:
            return
            
        u = int(msg.x)
        v = int(msg.y)
        
        # Get depth value (in mm for RealSense, convert to meters)
        try:
            depth_mm = self.latest_depth[v, u]
            if depth_mm == 0 or np.isnan(depth_mm):
                return
            z = depth_mm / 1000.0
        except IndexError:
            return
            
        # Compute 3D point in camera frame
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        
        # Create PoseStamped
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = self.depth_frame_id # Usually camera_depth_optical_frame
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.w = 1.0
        
        # Transform to 'world' frame using TF2
        try:
            transformed_pose = self.tf_buffer.transform(pose, 'world', timeout=rclpy.duration.Duration(seconds=1.0))
            self.pose_pub.publish(transformed_pose)
        except tf2_ros.Exception as e:
            self.get_logger().warn(f"TF Transform failed: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = DepthLocalizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
