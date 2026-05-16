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
        
        # Configuration
        self.declare_parameter('depth_window_size', 5)  # Window for median filtering
        self.declare_parameter('max_depth', 3.0)  # Max valid depth in meters
        self.depth_window = self.get_parameter('depth_window_size').value
        self.max_depth = self.get_parameter('max_depth').value

        # Publishers
        self.pose_pub = self.create_publisher(PoseStamped, '/object_pose', 10)

        # Subscribers
        self.centroid_sub = self.create_subscription(
            Pose2D, '/detected_object', self.centroid_callback, 10
        )
        self.depth_sub = self.create_subscription(
            Image, '/realsense/depth/image_rect_raw', self.depth_callback, 10
        )
        self.info_sub = self.create_subscription(
            CameraInfo, '/realsense/color/camera_info', self.info_callback, 10
        )

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.latest_depth = None
        self.depth_frame_id = 'camera_depth_optical_frame'  # Default, will be updated

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

    def get_median_depth(self, u: int, v: int, depth_img: np.ndarray) -> float | None:
        """
        Get median depth in a window around (u, v) to reduce noise.
        
        Returns depth in meters, or None if no valid readings.
        """
        h, w = depth_img.shape
        half_win = self.depth_window // 2
        
        # Define window bounds
        y_min = max(0, v - half_win)
        y_max = min(h, v + half_win + 1)
        x_min = max(0, u - half_win)
        x_max = min(w, u + half_win + 1)
        
        # Extract window
        window = depth_img[y_min:y_max, x_min:x_max]
        
        # Filter out invalid depths (0 or NaN)
        valid_depths = window[(window > 0) & (~np.isnan(window))]
        
        if len(valid_depths) == 0:
            return None
        
        # Take median for robustness
        median_depth = np.median(valid_depths)
        
        # Validate depth range
        if median_depth > self.max_depth:
            return None
        
        return float(median_depth)

    def centroid_callback(self, msg):
        if self.fx is None or self.latest_depth is None:
            self.get_logger().debug("Waiting for camera info and depth image...")
            return
            
        u = int(msg.x)
        v = int(msg.y)

        # Validate pixel coordinates
        h, w = self.latest_depth.shape
        if u < 0 or u >= w or v < 0 or v >= h:
            self.get_logger().warn(f"Centroid ({u}, {v}) outside image bounds ({w}x{h})")
            return

        # Get filtered depth value (in meters)
        depth_m = self.get_median_depth(u, v, self.latest_depth)
        
        if depth_m is None:
            self.get_logger().debug(f"No valid depth at ({u}, {v})")
            return
            
        z = depth_m

        # Compute 3D point in camera frame
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        
        # Create PoseStamped
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = self.depth_frame_id
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.w = 1.0
        
        # Transform to 'world' frame using TF2
        try:
            transformed_pose = self.tf_buffer.transform(
                pose, 'world', timeout=rclpy.duration.Duration(seconds=1.0)
            )
            self.pose_pub.publish(transformed_pose)
            self.get_logger().debug(f"Published object pose in world: x={x:.3f}, y={y:.3f}, z={z:.3f}")
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
