#!/usr/bin/env python3
"""
Runtime Calibration Node:
Detects an ArUco marker to compute the transform from Camera Frame -> World Frame.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import cv2.aruco as aruco
import numpy as np
from tf2_ros import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
import tf_transformations

class CalibrationNode(Node):
    def __init__(self):
        super().__init__('calibration_node')
        self.bridge = CvBridge()
        self.broadcaster = StaticTransformBroadcaster(self)
        
        # Configuration
        self.declare_parameter('aruco_dict', 4) # DICT_6X6_250
        self.declare_parameter('aruco_marker_id', 0)
        self.declare_parameter('marker_size_m', 0.05) # Size of the marker in meters (adjust based on print)
        
        # Marker position relative to "world" frame (base_link). 
        # You measure this once with a ruler. X forward, Y left, Z up.
        self.declare_parameter('marker_world_x', 0.15)
        self.declare_parameter('marker_world_y', 0.0)
        self.declare_parameter('marker_world_z', 0.15) # Height above table
        
        # Subscribers (match the RealSense node name 'realsense' from launch file)
        self.image_sub = self.create_subscription(
            Image, '/realsense/color/image_raw', self.image_callback, 10)
        self.info_sub = self.create_subscription(
            CameraInfo, '/realsense/color/camera_info', self.info_callback, 10)
            
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibrated = False
        
        self.get_logger().info("Calibration node started. Show ArUco marker to camera to calibrate.")

    def info_callback(self, msg):
        self.camera_matrix = np.array(msg.k).reshape((3, 3))
        self.dist_coeffs = np.array(msg.d)
        
    def image_callback(self, msg):
        if self.calibrated:
            return

        if self.camera_matrix is None:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            return

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        parameters = aruco.DetectorParameters_create()
        
        corners, ids, rejected = aruco.detectMarkers(gray, dictionary, parameters=parameters)
        
        if ids is not None:
            target_id = self.get_parameter('aruco_marker_id').value
            if target_id in ids:
                idx = np.where(ids == target_id)[0][0]
                corner = corners[idx]
                
                marker_size = self.get_parameter('marker_size_m').value
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corner, marker_size, self.camera_matrix, self.dist_coeffs)
                
                self.broadcast_transform(rvec[0], tvec[0])
                self.calibrated = True
                self.destroy_node() # Stop running after successful calibration

    def broadcast_transform(self, rvec, tvec):
        # We have the marker position in the camera frame (tvec) and rotation (rvec)
        # We know the marker position in the world frame (from parameters)
        # We want to find the transform: World -> Camera
        
        # 1. Get Marker Pose in World frame
        mx, my, mz = self.get_parameter('marker_world_x').value, self.get_parameter('marker_world_y').value, self.get_parameter('marker_world_z').value
        # Assuming marker is upright on table (90 deg pitch) - adjust if needed
        marker_world_quat = tf_transformations.quaternion_from_euler(0, 1.5708, 0) 
        
        # 2. Get Camera Pose relative to Marker
        # rvec, tvec describes Marker relative to Camera.
        # We want Camera relative to Marker.
        R_cam_marker = np.eye(3)
        cv2.Rodrigues(rvec, R_cam_marker)
        t_cam_marker = tvec.flatten()
        
        # Invert to get Marker in Camera frame -> Camera in Marker frame
        R_marker_cam = R_cam_marker.T
        t_marker_cam = -R_marker_cam @ t_cam_marker
        q_marker_cam = tf_transformations.quaternion_from_matrix(np.eye(4)) # Placeholder
        q_marker_cam = tf_transformations.quaternion_from_rotation_matrix(R_marker_cam)
        
        # 3. Compose: World -> Marker -> Camera
        # TransformStamped expects: parent_frame_id -> child_frame_id
        # We want to publish: child_frame_id (camera) relative to parent_frame_id (world)
        
        # Combine Quaternions: q_world_cam = q_world_marker * q_marker_cam
        q_world_marker = marker_world_quat
        q_world_cam = tf_transformations.quaternion_multiply(q_world_marker, q_marker_cam)
        
        # Combine Translation: t_world_cam = t_world_marker + R_world_marker * t_marker_cam
        R_world_marker = tf_transformations.rotation_matrix(1.5708, [0, 1, 0])[0:3, 0:3]
        t_world_marker = np.array([mx, my, mz])
        t_world_cam = t_world_marker + (R_world_marker @ t_marker_cam)
        
        # Broadcast
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'realsense_color_optical_frame'
        
        t.transform.translation.x = float(t_world_cam[0])
        t.transform.translation.y = float(t_world_cam[1])
        t.transform.translation.z = float(t_world_cam[2])
        t.transform.rotation.x = float(q_world_cam[0])
        t.transform.rotation.y = float(q_world_cam[1])
        t.transform.rotation.z = float(q_world_cam[2])
        t.transform.rotation.w = float(q_world_cam[3])
        
        self.broadcaster.sendTransform(t)
        self.get_logger().info(f"Calibration successful! Broadcasting World -> Camera transform.")
        self.get_logger().info(f"Translation: {t_world_cam}, Rotation (quat): {q_world_cam}")

def main(args=None):
    rclpy.init(args=args)
    node = CalibrationNode()
    rclpy.spin(node)
    rclpy.shutdown()
