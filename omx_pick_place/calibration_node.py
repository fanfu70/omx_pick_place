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
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


class CalibrationNode(Node):
    def __init__(self):
        super().__init__('calibration_node')
        self.bridge = CvBridge()
        self.broadcaster = StaticTransformBroadcaster(self)
        
        # Configuration
        self.declare_parameter('aruco_dict', 4)  # DICT_6X6_250
        self.declare_parameter('aruco_marker_id_1', 0)
        self.declare_parameter('aruco_marker_id_2', 1)
        self.declare_parameter('marker_size_m', 0.05)  # Size of the marker in meters
        
        # Marker 1 position and orientation in world frame
        self.declare_parameter('marker1_world_x', 0.15)
        self.declare_parameter('marker1_world_y', 0.0)
        self.declare_parameter('marker1_world_z', 0.15)  # Height above table
        self.declare_parameter('marker1_world_roll', 0.0)
        self.declare_parameter('marker1_world_pitch', 1.5708)  # 90 degrees - marker lying flat
        self.declare_parameter('marker1_world_yaw', 0.0)
        
        # Marker 2 position and orientation in world frame
        self.declare_parameter('marker2_world_x', 0.20)
        self.declare_parameter('marker2_world_y', 0.0)
        self.declare_parameter('marker2_world_z', 0.15)  # Height above table
        self.declare_parameter('marker2_world_roll', 0.0)
        self.declare_parameter('marker2_world_pitch', 1.5708)  # 90 degrees - marker lying flat
        self.declare_parameter('marker2_world_yaw', 0.0)

        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibrated = False
        
        # Output directory for visualization
        self.declare_parameter('output_dir', os.path.expanduser('~/calibration_results'))
        self.output_dir = self.get_parameter('output_dir').value
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.image_callback, 10
        )
        self.info_sub = self.create_subscription(
            CameraInfo, '/camera/camera/color/camera_info', self.info_callback, 10
        )

        self.get_logger().info("Calibration node started. Show ArUco marker to camera to calibrate.")

    def info_callback(self, msg):
        self.camera_matrix = np.array(msg.k).reshape((3, 3))
        self.dist_coeffs = np.array(msg.d)
        self.get_logger().info("Camera info received. Intrinsics loaded.")

    def image_callback(self, msg):
        if self.calibrated:
            return

        if self.camera_matrix is None:
            self.get_logger().debug("Waiting for camera info...")
            return

        try:
            # Handle different encodings
            encoding = msg.encoding
            if encoding in ['rgb8', 'bgr8']:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            else:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            return

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        parameters = aruco.DetectorParameters_create()
        
        corners, ids, rejected = aruco.detectMarkers(gray, dictionary, parameters=parameters)
        
        if ids is not None:
            target_id_1 = self.get_parameter('aruco_marker_id_1').value
            target_id_2 = self.get_parameter('aruco_marker_id_2').value
            
            # Check if both markers are detected
            ids_flat = ids.flatten()
            if target_id_1 in ids_flat and target_id_2 in ids_flat:
                marker_size = self.get_parameter('marker_size_m').value
                
                # Get corners for both markers
                idx_1 = np.where(ids == target_id_1)[0][0]
                idx_2 = np.where(ids == target_id_2)[0][0]
                corner_1 = corners[idx_1]
                corner_2 = corners[idx_2]
                
                # Estimate pose for each marker
                rvec_1, tvec_1, _ = aruco.estimatePoseSingleMarkers(
                    corner_1, marker_size, self.camera_matrix, self.dist_coeffs
                )
                rvec_2, tvec_2, _ = aruco.estimatePoseSingleMarkers(
                    corner_2, marker_size, self.camera_matrix, self.dist_coeffs
                )
                
                # Store for visualization
                self.last_rvec_1 = rvec_1[0]
                self.last_tvec_1 = tvec_1[0]
                self.last_corner_1 = corner_1
                self.last_rvec_2 = rvec_2[0]
                self.last_tvec_2 = tvec_2[0]
                self.last_corner_2 = corner_2
                
                self.broadcast_transform(rvec_1[0], tvec_1[0], rvec_2[0], tvec_2[0], cv_image)
                self.calibrated = True
                self.get_logger().info("Calibration successful with both markers! Shutting down.")

                # Clean shutdown instead of destroy_node() in callback
                # Set a flag and let main() handle shutdown
            else:
                detected = [int(i) for i in ids_flat]
                self.get_logger().debug(f"Detected markers: {detected}, waiting for both {target_id_1} and {target_id_2}")

    def broadcast_transform(self, rvec_1, tvec_1, rvec_2, tvec_2, cv_image=None):
        """Compute and broadcast World -> Camera transform using two markers."""
        # Save image for visualization
        self.last_cv_image = cv_image
        
        # === Process Marker 1 ===
        mx_1 = self.get_parameter('marker1_world_x').value
        my_1 = self.get_parameter('marker1_world_y').value
        mz_1 = self.get_parameter('marker1_world_z').value
        roll_1 = self.get_parameter('marker1_world_roll').value
        pitch_1 = self.get_parameter('marker1_world_pitch').value
        yaw_1 = self.get_parameter('marker1_world_yaw').value
        
        # === Process Marker 2 ===
        mx_2 = self.get_parameter('marker2_world_x').value
        my_2 = self.get_parameter('marker2_world_y').value
        mz_2 = self.get_parameter('marker2_world_z').value
        roll_2 = self.get_parameter('marker2_world_roll').value
        pitch_2 = self.get_parameter('marker2_world_pitch').value
        yaw_2 = self.get_parameter('marker2_world_yaw').value
        
        # Compute camera pose from each marker
        t_world_cam_1, R_world_cam_1 = self.compute_camera_pose_from_marker(
            rvec_1, tvec_1, mx_1, my_1, mz_1, roll_1, pitch_1, yaw_1, "Marker 1"
        )
        t_world_cam_2, R_world_cam_2 = self.compute_camera_pose_from_marker(
            rvec_2, tvec_2, mx_2, my_2, mz_2, roll_2, pitch_2, yaw_2, "Marker 2"
        )
        
        # Fuse: average the two poses
        t_world_cam_fused = (t_world_cam_1 + t_world_cam_2) / 2.0
        
        # For rotation, use nearest rotation interpolation
        R_world_cam_fused = self.interpolate_rotations(R_world_cam_1, R_world_cam_2)
        
        # Compute fused quaternion
        T_fused = np.eye(4)
        T_fused[:3, :3] = R_world_cam_fused
        T_fused[:3, 3] = t_world_cam_fused
        q_world_cam_fused = tf_transformations.quaternion_from_matrix(T_fused)
        
        # Log individual and fused results
        self.get_logger().info(f"Marker 1 -> Camera: T={t_world_cam_1}")
        self.get_logger().info(f"Marker 2 -> Camera: T={t_world_cam_2}")
        self.get_logger().info(f"Fused Camera: T={t_world_cam_fused}")
        
        # Broadcast fused transform
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'realsense_color_optical_frame'
        
        t.transform.translation.x = float(t_world_cam_fused[0])
        t.transform.translation.y = float(t_world_cam_fused[1])
        t.transform.translation.z = float(t_world_cam_fused[2])
        t.transform.rotation.x = float(q_world_cam_fused[0])
        t.transform.rotation.y = float(q_world_cam_fused[1])
        t.transform.rotation.z = float(q_world_cam_fused[2])
        t.transform.rotation.w = float(q_world_cam_fused[3])
        
        self.broadcaster.sendTransform(t)
        self.get_logger().info(f"Broadcasting fused World -> Camera transform.")
        self.get_logger().info(f"Translation: {t_world_cam_fused}, Rotation (quat): {q_world_cam_fused}")
        
        # Store individual poses for visualization
        self.t_world_cam_1 = t_world_cam_1
        self.R_world_cam_1 = R_world_cam_1
        self.t_world_cam_2 = t_world_cam_2
        self.R_world_cam_2 = R_world_cam_2
        self.t_world_cam_fused = t_world_cam_fused
        self.R_world_cam_fused = R_world_cam_fused
        
        # Compute marker poses in world frame for visualization
        R_world_marker_1 = tf_transformations.euler_matrix(roll_1, pitch_1, yaw_1)[:3, :3]
        t_world_marker_1 = np.array([mx_1, my_1, mz_1])
        R_world_marker_2 = tf_transformations.euler_matrix(roll_2, pitch_2, yaw_2)[:3, :3]
        t_world_marker_2 = np.array([mx_2, my_2, mz_2])
        
        # Generate visualization of all frames
        self.visualize_poses(
            t_world_cam_fused, R_world_cam_fused,
            t_world_marker_1, R_world_marker_1, t_world_marker_2, R_world_marker_2
        )
        
        # Generate visualization of World -> Marker 1
        self.visualize_world_to_marker(t_world_marker_1, R_world_marker_1, "Marker 1")
        
        # Generate visualization of World -> Marker 2
        self.visualize_world_to_marker(t_world_marker_2, R_world_marker_2, "Marker 2")
        
        # Generate visualization of Marker 1 -> Camera
        R_cam_marker_1 = np.eye(3)
        cv2.Rodrigues(rvec_1, R_cam_marker_1)
        self.visualize_marker_to_camera(tvec_1.flatten(), R_cam_marker_1, "Marker 1")
        
        # Generate visualization of Marker 2 -> Camera
        R_cam_marker_2 = np.eye(3)
        cv2.Rodrigues(rvec_2, R_cam_marker_2)
        self.visualize_marker_to_camera(tvec_2.flatten(), R_cam_marker_2, "Marker 2")
        
        # Generate visualization of World -> Camera (fused transform)
        self.visualize_world_to_camera(t_world_cam_fused, R_world_cam_fused)
        
        # Generate camera image with overlaid both marker poses
        if self.last_cv_image is not None:
            self.visualize_camera_with_markers(
                self.last_cv_image,
                self.last_corner_1, self.last_rvec_1, self.last_tvec_1, "Marker 1",
                self.last_corner_2, self.last_rvec_2, self.last_tvec_2, "Marker 2"
            )

    def visualize_poses(self, t_world_cam, R_world_cam, t_world_marker_1, R_world_marker_1, t_world_marker_2, R_world_marker_2):
        """Generate a 3D visualization of World, Camera, and both Marker frames."""
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        axis_length = 0.15
        
        # World frame (origin)
        world_pos = np.array([0.0, 0.0, 0.0])
        self.plot_frame_with_label(ax, world_pos, np.eye(3), 'World', axis_length=axis_length)
        
        # Camera frame in world coordinates (fused)
        camera_pos = t_world_cam
        self.plot_frame_with_label(ax, camera_pos, R_world_cam, 'Camera (fused)', axis_length=axis_length)
        
        # Marker 1 frame in world coordinates
        self.plot_frame_with_label(ax, t_world_marker_1, R_world_marker_1, 'Marker 1', axis_length=axis_length)
        
        # Marker 2 frame in world coordinates
        self.plot_frame_with_label(ax, t_world_marker_2, R_world_marker_2, 'Marker 2', axis_length=axis_length)
        
        # Draw connections
        ax.plot([world_pos[0], camera_pos[0]], 
                [world_pos[1], camera_pos[1]], 
                [world_pos[2], camera_pos[2]], 'k--', alpha=0.4, linewidth=1.5, label='World → Camera')
        ax.plot([camera_pos[0], t_world_marker_1[0]], 
                [camera_pos[1], t_world_marker_1[1]], 
                [camera_pos[2], t_world_marker_1[2]], 'b--', alpha=0.4, linewidth=1.5, label='Camera → Marker 1')
        ax.plot([camera_pos[0], t_world_marker_2[0]], 
                [camera_pos[1], t_world_marker_2[1]], 
                [camera_pos[2], t_world_marker_2[2]], 'g--', alpha=0.4, linewidth=1.5, label='Camera → Marker 2')
        
        # Set labels and title
        ax.set_xlabel('X (m)', fontsize=12, labelpad=10)
        ax.set_ylabel('Y (m)', fontsize=12, labelpad=10)
        ax.set_zlabel('Z (m)', fontsize=12, labelpad=10)
        ax.set_title('Calibration: World, Camera, and Two Marker Frames', fontsize=14, pad=20)
        ax.legend(loc='upper left', fontsize=10)
        
        # Set axis range
        axis_limit = 0.8
        ax.set_xlim([-axis_limit, axis_limit])
        ax.set_ylim([-axis_limit, axis_limit])
        ax.set_zlim([-axis_limit, axis_limit])
        
        # Save figure
        output_path = os.path.join(self.output_dir, 'calibration_poses.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.get_logger().info(f"Visualization saved to: {output_path}")

    def visualize_world_to_marker(self, t_world_marker, R_world_marker, marker_name="Marker"):
        """Generate a 3D visualization of World -> Marker transform to verify input parameters."""
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        axis_length = 0.15
        
        # World frame (origin)
        world_pos = np.array([0.0, 0.0, 0.0])
        self.plot_frame_with_label(ax, world_pos, np.eye(3), 'World', axis_length=axis_length)
        
        # Marker frame in world coordinates (from input parameters)
        marker_pos = t_world_marker
        self.plot_frame_with_label(ax, marker_pos, R_world_marker, marker_name, axis_length=axis_length)
        
        # Draw connection
        ax.plot([world_pos[0], marker_pos[0]], 
                [world_pos[1], marker_pos[1]], 
                [world_pos[2], marker_pos[2]], 'k--', alpha=0.4, linewidth=1.5, 
                label=f'World → {marker_name} (input params)')
        
        # Add parameter info as text
        mx, my, mz = marker_pos
        T_marker = np.eye(4)
        T_marker[:3, :3] = R_world_marker
        euler = tf_transformations.euler_from_matrix(T_marker)
        roll, pitch, yaw = euler[0], euler[1], euler[2]
        
        param_text = (f"{marker_name} Input Parameters:\n"
                     f"Position: ({mx:.3f}, {my:.3f}, {mz:.3f}) m\n"
                     f"Roll: {np.degrees(roll):.1f}°\n"
                     f"Pitch: {np.degrees(pitch):.1f}°\n"
                     f"Yaw: {np.degrees(yaw):.1f}°")
        
        ax.text2D(0.02, 0.98, param_text, transform=ax.transAxes, fontsize=10,
                  verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Set labels and title
        ax.set_xlabel('X (m)', fontsize=12, labelpad=10)
        ax.set_ylabel('Y (m)', fontsize=12, labelpad=10)
        ax.set_zlabel('Z (m)', fontsize=12, labelpad=10)
        ax.set_title(f'World → {marker_name} Transform (Input Parameter Verification)', fontsize=14, pad=20)
        ax.legend(loc='upper left', fontsize=10)
        
        # Set axis range from -0.8m to 0.8m
        axis_limit = 0.8
        ax.set_xlim([-axis_limit, axis_limit])
        ax.set_ylim([-axis_limit, axis_limit])
        ax.set_zlim([-axis_limit, axis_limit])
        
        # Save figure
        output_path = os.path.join(self.output_dir, f'world_to_{marker_name.lower().replace(" ", "_")}.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.get_logger().info(f"World->{marker_name} visualization saved to: {output_path}")

    def visualize_marker_to_camera(self, t_cam_marker, R_cam_marker, marker_name="Marker"):
        """Generate a 3D visualization of Marker -> Camera transform from ArUco detection."""
        fig = plt.figure(figsize=(16, 12))
        
        axis_length = 0.15
        
        # === Left plot: Raw ArUco result (Marker in Camera frame) ===
        ax1 = fig.add_subplot(121, projection='3d')
        
        # Camera frame (origin)
        camera_pos_raw = np.array([0.0, 0.0, 0.0])
        self.plot_frame_with_label(ax1, camera_pos_raw, np.eye(3), 'Camera (origin)', axis_length=axis_length)
        
        # Marker in Camera frame (raw ArUco output)
        marker_pos_raw = t_cam_marker
        self.plot_frame_with_label(ax1, marker_pos_raw, R_cam_marker, marker_name, axis_length=axis_length)
        
        # Draw connection
        ax1.plot([camera_pos_raw[0], marker_pos_raw[0]], 
                [camera_pos_raw[1], marker_pos_raw[1]], 
                [camera_pos_raw[2], marker_pos_raw[2]], 'k--', alpha=0.4, linewidth=1.5)
        
        mx, my, mz = marker_pos_raw
        raw_text = (f"Raw ArUco Output ({marker_name}):\n"
                   f"Marker in Camera Frame:\n"
                   f"  Position: ({mx:.3f}, {my:.3f}, {mz:.3f}) m\n"
                   f"Camera convention:\n"
                   f"  X=right, Y=down, Z=forward")
        ax1.text2D(0.5, 0.98, raw_text, transform=ax1.transAxes, fontsize=9,
                  horizontalalignment='center', verticalalignment='top', 
                  bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        ax1.set_xlabel('X (m)', fontsize=10)
        ax1.set_ylabel('Y (m)', fontsize=10)
        ax1.set_zlabel('Z (m)', fontsize=10)
        ax1.set_title(f'Raw ArUco: {marker_name} in Camera Frame', fontsize=12, pad=10)
        
        axis_limit = 0.8
        ax1.set_xlim([-axis_limit, axis_limit])
        ax1.set_ylim([-axis_limit, axis_limit])
        ax1.set_zlim([-axis_limit, axis_limit])
        
        # === Right plot: Inverted (Camera in Marker frame) ===
        ax2 = fig.add_subplot(122, projection='3d')
        
        # Invert the transform: Camera pose in Marker frame
        R_marker_cam = R_cam_marker.T
        t_marker_cam = -R_marker_cam @ t_cam_marker
        
        # Marker frame (origin)
        marker_pos = np.array([0.0, 0.0, 0.0])
        self.plot_frame_with_label(ax2, marker_pos, np.eye(3), f'{marker_name} (origin)', axis_length=axis_length)
        
        # Camera in Marker frame (inverted)
        camera_pos = t_marker_cam
        self.plot_frame_with_label(ax2, camera_pos, R_marker_cam, 'Camera', axis_length=axis_length)
        
        # Draw connection
        ax2.plot([marker_pos[0], camera_pos[0]], 
                [marker_pos[1], camera_pos[1]], 
                [marker_pos[2], camera_pos[2]], 'k--', alpha=0.4, linewidth=1.5)
        
        cx, cy, cz = camera_pos
        T_cam = np.eye(4)
        T_cam[:3, :3] = R_marker_cam
        euler = tf_transformations.euler_from_matrix(T_cam)
        roll, pitch, yaw = euler[0], euler[1], euler[2]
        
        inv_text = (f"Inverted Transform ({marker_name}):\n"
                   f"Camera in {marker_name} Frame:\n"
                   f"  Position: ({cx:.3f}, {cy:.3f}, {cz:.3f}) m\n"
                   f"  Roll: {np.degrees(roll):.1f}°\n"
                   f"  Pitch: {np.degrees(pitch):.1f}°\n"
                   f"  Yaw: {np.degrees(yaw):.1f}°")
        ax2.text2D(0.5, 0.98, inv_text, transform=ax2.transAxes, fontsize=9,
                  horizontalalignment='center', verticalalignment='top', 
                  bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        ax2.set_xlabel('X (m)', fontsize=10)
        ax2.set_ylabel('Y (m)', fontsize=10)
        ax2.set_zlabel('Z (m)', fontsize=10)
        ax2.set_title(f'Inverted: Camera in {marker_name} Frame', fontsize=12, pad=10)
        
        ax2.set_xlim([-axis_limit, axis_limit])
        ax2.set_ylim([-axis_limit, axis_limit])
        ax2.set_zlim([-axis_limit, axis_limit])
        
        # Save figure
        output_path = os.path.join(self.output_dir, f'{marker_name.lower().replace(" ", "_")}_to_camera.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.get_logger().info(f"{marker_name}->Camera visualization saved to: {output_path}")

    def visualize_world_to_camera(self, t_world_cam, R_world_cam):
        """Generate a 3D visualization of World -> Camera transform (composed result)."""
        fig = plt.figure(figsize=(16, 12))
        
        axis_length = 0.15
        
        # === Left plot: World -> Camera (Camera in World frame) ===
        ax1 = fig.add_subplot(121, projection='3d')
        
        # World frame (origin)
        world_pos = np.array([0.0, 0.0, 0.0])
        self.plot_frame_with_label(ax1, world_pos, np.eye(3), 'World (origin)', axis_length=axis_length)
        
        # Camera in World frame (composed transform)
        camera_pos = t_world_cam
        self.plot_frame_with_label(ax1, camera_pos, R_world_cam, 'Camera', axis_length=axis_length)
        
        # Draw connection
        ax1.plot([world_pos[0], camera_pos[0]], 
                [world_pos[1], camera_pos[1]], 
                [world_pos[2], camera_pos[2]], 'k--', alpha=0.4, linewidth=1.5)
        
        cx, cy, cz = camera_pos
        T_cam = np.eye(4)
        T_cam[:3, :3] = R_world_cam
        euler = tf_transformations.euler_from_matrix(T_cam)
        roll, pitch, yaw = euler[0], euler[1], euler[2]
        
        comp_text = (f"Composed Transform:\n"
                    f"Camera in World Frame:\n"
                    f"  Position: ({cx:.3f}, {cy:.3f}, {cz:.3f}) m\n"
                    f"  Roll: {np.degrees(roll):.1f}°\n"
                    f"  Pitch: {np.degrees(pitch):.1f}°\n"
                    f"  Yaw: {np.degrees(yaw):.1f}°")
        ax1.text2D(0.5, 0.98, comp_text, transform=ax1.transAxes, fontsize=9,
                  horizontalalignment='center', verticalalignment='top', 
                  bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        ax1.set_xlabel('X (m)', fontsize=10)
        ax1.set_ylabel('Y (m)', fontsize=10)
        ax1.set_zlabel('Z (m)', fontsize=10)
        ax1.set_title('World → Camera: Camera in World Frame', fontsize=12, pad=10)
        
        axis_limit = 0.8
        ax1.set_xlim([-axis_limit, axis_limit])
        ax1.set_ylim([-axis_limit, axis_limit])
        ax1.set_zlim([-axis_limit, axis_limit])
        
        # === Right plot: Inverted (World in Camera frame) ===
        ax2 = fig.add_subplot(122, projection='3d')
        
        # Invert: World in Camera frame
        R_cam_world = R_world_cam.T
        t_cam_world = -R_cam_world @ t_world_cam
        
        # Camera frame (origin)
        camera_pos_inv = np.array([0.0, 0.0, 0.0])
        self.plot_frame_with_label(ax2, camera_pos_inv, np.eye(3), 'Camera (origin)', axis_length=axis_length)
        
        # World in Camera frame (inverted)
        world_pos_inv = t_cam_world
        self.plot_frame_with_label(ax2, world_pos_inv, R_cam_world, 'World', axis_length=axis_length)
        
        # Draw connection
        ax2.plot([camera_pos_inv[0], world_pos_inv[0]], 
                [camera_pos_inv[1], world_pos_inv[1]], 
                [camera_pos_inv[2], world_pos_inv[2]], 'k--', alpha=0.4, linewidth=1.5)
        
        wx, wy, wz = world_pos_inv
        T_world = np.eye(4)
        T_world[:3, :3] = R_cam_world
        euler_inv = tf_transformations.euler_from_matrix(T_world)
        roll_inv, pitch_inv, yaw_inv = euler_inv[0], euler_inv[1], euler_inv[2]
        
        inv_text = (f"Inverted Transform:\n"
                   f"World in Camera Frame:\n"
                   f"  Position: ({wx:.3f}, {wy:.3f}, {wz:.3f}) m\n"
                   f"  Roll: {np.degrees(roll_inv):.1f}°\n"
                   f"  Pitch: {np.degrees(pitch_inv):.1f}°\n"
                   f"  Yaw: {np.degrees(yaw_inv):.1f}°")
        ax2.text2D(0.5, 0.98, inv_text, transform=ax2.transAxes, fontsize=9,
                  horizontalalignment='center', verticalalignment='top', 
                  bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        ax2.set_xlabel('X (m)', fontsize=10)
        ax2.set_ylabel('Y (m)', fontsize=10)
        ax2.set_zlabel('Z (m)', fontsize=10)
        ax2.set_title('Inverted: World in Camera Frame', fontsize=12, pad=10)
        
        ax2.set_xlim([-axis_limit, axis_limit])
        ax2.set_ylim([-axis_limit, axis_limit])
        ax2.set_zlim([-axis_limit, axis_limit])
        
        # Save figure
        output_path = os.path.join(self.output_dir, 'world_to_camera.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.get_logger().info(f"World->Camera visualization saved to: {output_path}")

    def visualize_camera_with_marker(self, cv_image, corner, rvec, tvec):
        """Generate a PNG of the camera image with overlaid ArUco marker pose axes."""
        # Make a copy of the image to draw on
        img_overlay = cv_image.copy()
        
        # Draw detected marker contours
        aruco.drawDetectedMarkers(img_overlay, [corner])
        
        # Draw the marker pose axes manually (cv2.aruco.drawAxis not available in all versions)
        marker_size = self.get_parameter('marker_size_m').value
        axis_length = marker_size  # Use marker size as axis length for reference
        
        # Project 3D axis points to 2D image coordinates
        # The rvec/tvec from ArUco describe: Camera -> Marker transform
        # So we project points in the marker frame
        # X axis (red): (axis_length, 0, 0) in marker frame
        # Y axis (green): (0, axis_length, 0) in marker frame
        # Z axis (blue): (0, 0, axis_length) in marker frame
        axis_points_3d = np.array([
            [axis_length, 0, 0],   # X axis end (Red)
            [0, axis_length, 0],   # Y axis end (Green)
            [0, 0, axis_length]    # Z axis end (Blue)
        ], dtype=np.float32).reshape(-1, 1, 3)
        
        # Project using the marker pose (rvec, tvec are Camera -> Marker)
        # cv2.projectPoints projects 3D object points to 2D image
        axis_points_2d, _ = cv2.projectPoints(axis_points_3d, rvec, tvec, 
                                               self.camera_matrix, self.dist_coeffs)
        
        # Get the marker center (origin of marker frame)
        center_3d = np.array([[0, 0, 0]], dtype=np.float32).reshape(-1, 1, 3)
        center_2d, _ = cv2.projectPoints(center_3d, rvec, tvec, 
                                          self.camera_matrix, self.dist_coeffs)
        center_pt = tuple(map(int, center_2d[0][0]))
        
        # Draw axes with correct colors (BGR format)
        # In BGR: (Blue, Green, Red)
        # X=Red: (0, 0, 255) -> Blue=0, Green=0, Red=255
        # Y=Green: (0, 255, 0) -> Blue=0, Green=255, Red=0
        # Z=Blue: (255, 0, 0) -> Blue=255, Green=0, Red=0
        colors_bgr = [
            (0, 0, 255),     # Red for X axis
            (0, 255, 0),     # Green for Y axis
            (255, 0, 0),     # Blue for Z axis
        ]
        axis_labels = ['X', 'Y', 'Z']
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        
        for i, (point_2d, color, label) in enumerate(zip(axis_points_2d, colors_bgr, axis_labels)):
            end_pt = tuple(map(int, point_2d[0]))
            # Draw axis line
            cv2.line(img_overlay, center_pt, end_pt, color, 3)
            # Draw arrow head
            direction = np.array(end_pt, dtype=float) - np.array(center_pt, dtype=float)
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
                arrow_length = 10
                arrow_angle = np.pi / 6  # 30 degrees in radians
                
                angle = np.arctan2(direction[1], direction[0])
                p1 = np.array(end_pt) - arrow_length * np.array([np.cos(angle + arrow_angle), 
                                                                   np.sin(angle + arrow_angle)])
                p2 = np.array(end_pt) - arrow_length * np.array([np.cos(angle - arrow_angle), 
                                                                   np.sin(angle - arrow_angle)])
                cv2.line(img_overlay, tuple(map(int, p1)), end_pt, color, 3)
                cv2.line(img_overlay, tuple(map(int, p2)), end_pt, color, 3)
                
                # Draw axis label
                label_pos = tuple(map(int, np.array(end_pt) + 10 * direction))
                cv2.putText(img_overlay, label, label_pos, font, font_scale, color, font_thickness, cv2.LINE_AA)
        
        # Add text overlay with pose information
        t_cam_marker = tvec.flatten()
        text_lines = [
            f"Marker in Camera Frame:",
            f"  Position: ({t_cam_marker[0]:.3f}, {t_cam_marker[1]:.3f}, {t_cam_marker[2]:.3f}) m",
        ]
        
        # Calculate rotation angles from rvec
        R_cam_marker = np.eye(3)
        cv2.Rodrigues(rvec, R_cam_marker)
        T_cam = np.eye(4)
        T_cam[:3, :3] = R_cam_marker
        euler = tf_transformations.euler_from_matrix(T_cam)
        roll, pitch, yaw = np.degrees(euler[0]), np.degrees(euler[1]), np.degrees(euler[2])
        text_lines.extend([
            f"  Roll: {roll:.1f}°, Pitch: {pitch:.1f}°, Yaw: {yaw:.1f}°",
        ])
        
        # Draw text on image
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        text_color = (0, 255, 0)  # Green in BGR
        
        # Get image dimensions
        h, w = img_overlay.shape[:2]
        
        # Calculate text position (top-left corner with padding)
        padding = 10
        line_height = int(20 * font_scale) + 5
        text_width = int(w * 0.35)
        text_height = len(text_lines) * line_height + padding * 2
        
        # Draw semi-transparent background
        overlay_rect = img_overlay[0:text_height, 0:text_width].copy()
        overlay_rect = cv2.addWeighted(overlay_rect, 0.3, np.zeros_like(overlay_rect), 0.7, 0)
        img_overlay[0:text_height, 0:text_width] = overlay_rect
        
        # Draw each line of text
        for i, line in enumerate(text_lines):
            y_pos = padding + 20 + i * line_height
            cv2.putText(img_overlay, line, (padding, y_pos), font, font_scale, 
                       text_color, font_thickness, cv2.LINE_AA)
        
        # Save the image
        output_path = os.path.join(self.output_dir, 'camera_with_marker_pose.png')
        cv2.imwrite(output_path, img_overlay)
        
        # Save the image
        output_path = os.path.join(self.output_dir, 'camera_with_marker_pose.png')
        cv2.imwrite(output_path, img_overlay)
        
        self.get_logger().info(f"Camera image with marker pose saved to: {output_path}")

    def visualize_camera_with_markers(self, cv_image, corner_1, rvec_1, tvec_1, name_1, 
                                       corner_2, rvec_2, tvec_2, name_2):
        """Generate a PNG of the camera image with overlaid both marker pose axes."""
        # Make a copy of the image to draw on
        img_overlay = cv_image.copy()
        
        # Draw both detected marker contours
        aruco.drawDetectedMarkers(img_overlay, [corner_1, corner_2])
        
        marker_size = self.get_parameter('marker_size_m').value
        axis_length = marker_size
        
        # Define colors for each marker
        marker_colors = {
            name_1: [(0, 0, 255), (0, 255, 0), (255, 0, 0)],  # Red, Green, Blue for X, Y, Z
            name_2: [(255, 0, 255), (0, 255, 255), (255, 255, 0)],  # Magenta, Cyan, Yellow for X, Y, Z
        }
        
        # Draw axes for both markers
        for corner, rvec, tvec, name in [(corner_1, rvec_1, tvec_1, name_1), 
                                          (corner_2, rvec_2, tvec_2, name_2)]:
            # Project 3D axis points to 2D image coordinates
            axis_points_3d = np.array([
                [axis_length, 0, 0],   # X axis end
                [0, axis_length, 0],   # Y axis end
                [0, 0, axis_length]    # Z axis end
            ], dtype=np.float32).reshape(-1, 1, 3)
            
            # Project using the marker pose
            axis_points_2d, _ = cv2.projectPoints(axis_points_3d, rvec, tvec, 
                                                   self.camera_matrix, self.dist_coeffs)
            
            # Get the marker center
            center_3d = np.array([[0, 0, 0]], dtype=np.float32).reshape(-1, 1, 3)
            center_2d, _ = cv2.projectPoints(center_3d, rvec, tvec, 
                                              self.camera_matrix, self.dist_coeffs)
            center_pt = tuple(map(int, center_2d[0][0]))
            
            colors_bgr = marker_colors[name]
            axis_labels = ['X', 'Y', 'Z']
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            font_thickness = 2
            
            for i, (point_2d, color, label) in enumerate(zip(axis_points_2d, colors_bgr, axis_labels)):
                end_pt = tuple(map(int, point_2d[0]))
                # Draw axis line
                cv2.line(img_overlay, center_pt, end_pt, color, 3)
                # Draw arrow head
                direction = np.array(end_pt, dtype=float) - np.array(center_pt, dtype=float)
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction = direction / norm
                    arrow_length = 10
                    arrow_angle = np.pi / 6
                    
                    angle = np.arctan2(direction[1], direction[0])
                    p1 = np.array(end_pt) - arrow_length * np.array([np.cos(angle + arrow_angle), 
                                                                       np.sin(angle + arrow_angle)])
                    p2 = np.array(end_pt) - arrow_length * np.array([np.cos(angle - arrow_angle), 
                                                                       np.sin(angle - arrow_angle)])
                    cv2.line(img_overlay, tuple(map(int, p1)), end_pt, color, 3)
                    cv2.line(img_overlay, tuple(map(int, p2)), end_pt, color, 3)
                    
                    # Draw axis label
                    label_pos = tuple(map(int, np.array(end_pt) + 10 * direction))
                    cv2.putText(img_overlay, label, label_pos, font, font_scale, color, font_thickness, cv2.LINE_AA)
            
            # Draw marker name near center
            cv2.putText(img_overlay, name, (center_pt[0] + 15, center_pt[1] - 15), 
                       font, 0.6, colors_bgr[0], 2, cv2.LINE_AA)
        
        # Add text overlay with pose information for both markers
        text_lines = ["Detected Markers:"]
        for rvec, tvec, name in [(rvec_1, tvec_1, name_1), (rvec_2, tvec_2, name_2)]:
            t_cam_marker = tvec.flatten()
            R_cam_marker = np.eye(3)
            cv2.Rodrigues(rvec, R_cam_marker)
            T_cam = np.eye(4)
            T_cam[:3, :3] = R_cam_marker
            euler = tf_transformations.euler_from_matrix(T_cam)
            roll, pitch, yaw = np.degrees(euler[0]), np.degrees(euler[1]), np.degrees(euler[2])
            text_lines.extend([
                f"{name}:",
                f"  T: ({t_cam_marker[0]:.3f}, {t_cam_marker[1]:.3f}, {t_cam_marker[2]:.3f}) m",
                f"  R: {roll:.1f}°, P: {pitch:.1f}°, Y: {yaw:.1f}°",
            ])
        
        # Draw text on image
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        text_color = (0, 255, 0)  # Green in BGR
        
        h, w = img_overlay.shape[:2]
        padding = 10
        line_height = int(20 * font_scale) + 5
        text_width = int(w * 0.4)
        text_height = len(text_lines) * line_height + padding * 2
        
        # Draw semi-transparent background
        overlay_rect = img_overlay[0:text_height, 0:text_width].copy()
        overlay_rect = cv2.addWeighted(overlay_rect, 0.3, np.zeros_like(overlay_rect), 0.7, 0)
        img_overlay[0:text_height, 0:text_width] = overlay_rect
        
        # Draw each line of text
        for i, line in enumerate(text_lines):
            y_pos = padding + 20 + i * line_height
            cv2.putText(img_overlay, line, (padding, y_pos), font, font_scale, 
                       text_color, font_thickness, cv2.LINE_AA)
        
        # Save the image
        output_path = os.path.join(self.output_dir, 'camera_with_markers_pose.png')
        cv2.imwrite(output_path, img_overlay)
        
        self.get_logger().info(f"Camera image with both markers saved to: {output_path}")

    def compute_camera_pose_from_marker(self, rvec, tvec, mx, my, mz, roll, pitch, yaw, marker_name):
        """Compute camera pose in world frame from a single marker."""
        # Get Marker Pose in Camera frame
        R_cam_marker = np.eye(3)
        cv2.Rodrigues(rvec, R_cam_marker)
        t_cam_marker = tvec.flatten()
        
        # Invert to get Camera in Marker frame
        R_marker_cam = R_cam_marker.T
        t_marker_cam = -R_marker_cam @ t_cam_marker
        
        # Get Marker Pose in World frame
        R_world_marker = tf_transformations.euler_matrix(roll, pitch, yaw)[:3, :3]
        t_world_marker = np.array([mx, my, mz])
        
        # Compose: World -> Marker -> Camera
        R_world_cam = R_world_marker @ R_cam_marker
        t_world_cam = t_world_marker + (R_world_marker @ t_marker_cam)
        
        self.get_logger().info(f"{marker_name} - Camera in World: T={t_world_cam}")
        return t_world_cam, R_world_cam

    def interpolate_rotations(self, R1, R2):
        """Interpolate between two rotation matrices using quaternion slerp."""
        # Convert to quaternions (returns tuples, convert to numpy arrays)
        T1 = np.eye(4)
        T1[:3, :3] = R1
        q1 = np.array(tf_transformations.quaternion_from_matrix(T1))
        
        T2 = np.eye(4)
        T2[:3, :3] = R2
        q2 = np.array(tf_transformations.quaternion_from_matrix(T2))
        
        # Ensure shortest path
        if np.dot(q1, q2) < 0:
            q2 = -q2
        
        # Simple average (for small angles, this is close enough to slerp)
        q_sum = q1 + q2
        q_norm = np.linalg.norm(q_sum)
        if q_norm < 1e-10:
            # Quaternions are opposite, fall back to identity
            self.get_logger().warn("Quaternions are opposite, using identity rotation for interpolation.")
            return np.eye(3)
        q_avg = q_sum / q_norm
        
        # Convert back to rotation matrix
        R_avg = tf_transformations.quaternion_matrix(q_avg)[:3, :3]
        return R_avg

    def plot_frame_with_label(self, ax, origin, rotation_matrix, label, axis_length=0.1):
        """Plot a coordinate frame with text label at the given origin."""
        # X axis (red)
        x_end = origin + axis_length * rotation_matrix[:, 0]
        ax.plot([origin[0], x_end[0]], [origin[1], x_end[1]], [origin[2], x_end[2]], 
                'r-', linewidth=3)
        
        # Y axis (green)
        y_end = origin + axis_length * rotation_matrix[:, 1]
        ax.plot([origin[0], y_end[0]], [origin[1], y_end[1]], [origin[2], y_end[2]], 
                'g-', linewidth=3)
        
        # Z axis (blue)
        z_end = origin + axis_length * rotation_matrix[:, 2]
        ax.plot([origin[0], z_end[0]], [origin[1], z_end[1]], [origin[2], z_end[2]], 
                'b-', linewidth=3)
        
        # Add text label next to the frame
        offset = np.array([0.05, 0.05, 0.05])
        label_pos = origin + offset
        ax.text(label_pos[0], label_pos[1], label_pos[2], label, 
                fontsize=12, fontweight='bold', color='darkred')
        
        # Print pose info to log
        self.get_logger().info(f"{label} frame position: [{origin[0]:.3f}, {origin[1]:.3f}, {origin[2]:.3f}]")

    def plot_frame(self, ax, origin, rotation_matrix, label, color='black', axis_length=0.1):
        """Plot a coordinate frame at the given origin with the given orientation."""
        # X axis (red)
        x_end = origin + axis_length * rotation_matrix[:, 0]
        ax.plot([origin[0], x_end[0]], [origin[1], x_end[1]], [origin[2], x_end[2]], 
                'r-', linewidth=2, label=f'{label} X' if label else 'X')
        
        # Y axis (green)
        y_end = origin + axis_length * rotation_matrix[:, 1]
        ax.plot([origin[0], y_end[0]], [origin[1], y_end[1]], [origin[2], y_end[2]], 
                'g-', linewidth=2, label=f'{label} Y' if label else 'Y')
        
        # Z axis (blue)
        z_end = origin + axis_length * rotation_matrix[:, 2]
        ax.plot([origin[0], z_end[0]], [origin[1], z_end[1]], [origin[2], z_end[2]], 
                'b-', linewidth=2, label=f'{label} Z' if label else 'Z')


def main(args=None):
    rclpy.init(args=args)
    node = CalibrationNode()
    try:
        # Spin until calibrated
        while rclpy.ok() and not node.calibrated:
            rclpy.spin_once(node, timeout_sec=0.1)
        node.get_logger().info("Calibration node shutting down.")
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()