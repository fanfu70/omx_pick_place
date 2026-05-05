#!/usr/bin/env python3
"""
Pick & Place Orchestrator:
Listens for object pose, plans and executes pick/place trajectory using MoveIt 2.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Pose
from moveit_commander import MoveGroupCommander, RobotCommander, PlanningSceneInterface
import moveit_msgs.msg
import tf_transformations
import time

class PickPlaceNode(Node):
    def __init__(self):
        super().__init__('pick_place_node')
        
        # MoveIt Setup
        self.robot = RobotCommander()
        self.move_group = MoveGroupCommander("arm") # Group name might be 'arm' or 'manipulator'
        self.planning_scene = PlanningSceneInterface()
        
        self.move_group.set_planning_frame("world")
        self.move_group.set_end_effector_link("end_effector_link") # Adjust if your URDF is different
        self.move_group.set_max_velocity_scaling_factor(0.5)
        self.move_group.set_max_acceleration_scaling_factor(0.5)
        
        # Configuration
        self.declare_parameter('approach_dist', 0.15)
        self.declare_parameter('lift_dist', 0.20)
        self.declare_parameter('grasp_depth', 0.02)
        self.declare_parameter('place_x', -0.3)
        self.declare_parameter('place_y', 0.0)
        self.declare_parameter('place_z', 0.20)
        
        self.approach_dist = self.get_parameter('approach_dist').value
        self.lift_dist = self.get_parameter('lift_dist').value
        self.grasp_depth = self.get_parameter('grasp_depth').value
        
        # Gripper joint
        self.gripper_joint = 'gripper_left_joint'
        
        # Subscribers
        self.pose_sub = self.create_subscription(
            PoseStamped, '/object_pose', self.pose_callback, 10)
            
        self.get_logger().info("Pick-Place Orchestrator started. Waiting for object pose...")
        
        # Initialize gripper to open
        self.control_gripper(0.0) # 0.0 is typically open for DYNAMIXEL gripper

    def control_gripper(self, position):
        """Control the gripper joint directly via ros2_control joint trajectory or set_joint_value."""
        # Note: In MoveIt 2, we often use a separate group for the gripper or just joint values.
        # Since we only have one joint name, we'll use the robot commander to set it.
        joint_names = [self.gripper_joint]
        joint_values = [position]
        
        self.robot.get_group(self.move_group.get_name()).go(joint_values, wait=True)
        self.get_logger().info(f"Gripper set to {position}")

    def pose_callback(self, msg):
        self.get_logger().info(f"Object detected at: {msg.pose.position}")
        self.execute_pick_and_place(msg.pose.position)

    def execute_pick_and_place(self, obj_position):
        # Define waypoints (x, y, z)
        obj_x, obj_y, obj_z = obj_position.x, obj_position.y, obj_position.z
        place_x, place_y, place_z = self.get_parameter('place_x').value, self.get_parameter('place_y').value, self.get_parameter('place_z').value
        
        # Helper to create a pose with zero rotation
        def make_pose(p, rpy=(0,0,0)):
            p_msg = Pose()
            p_msg.position.x, p_msg.position.y, p_msg.position.z = p[0], p[1], p[2]
            q = tf_transformations.quaternion_from_euler(*rpy)
            p_msg.orientation.x, p_msg.orientation.y, p_msg.orientation.z, p_msg.orientation.w = q
            return p_msg
            
        # 1. Home / Pre-grasp (Above object)
        pre_grasp = make_pose((obj_x, obj_y, obj_z + self.approach_dist))
        self.move_group.set_pose_target(pre_grasp)
        self.go()
        
        # 2. Open Gripper
        self.control_gripper(0.0)
        
        # 3. Approach (Closer to object)
        approach = make_pose((obj_x, obj_y, obj_z + 0.03))
        self.move_group.set_pose_target(approach)
        self.go()
        
        # 4. Grasp (Slightly into object)
        grasp = make_pose((obj_x, obj_y, obj_z - self.grasp_depth))
        self.move_group.set_pose_target(grasp)
        self.go()
        
        # 5. Close Gripper
        self.control_gripper(0.8) # 0.8 is typically closed (adjust for your hardware)
        
        # 6. Lift
        lift = make_pose((obj_x, obj_y, obj_z + self.lift_dist))
        self.move_group.set_pose_target(lift)
        self.go()
        
        # 7. Move to Place
        place = make_pose((place_x, place_y, place_z))
        self.move_group.set_pose_target(place)
        self.go()
        
        # 8. Open Gripper
        self.control_gripper(0.0)
        
        # 9. Retreat (Lift up slightly after placing)
        retreat = make_pose((place_x, place_y, place_z + 0.1))
        self.move_group.set_pose_target(retreat)
        self.go()
        
        self.get_logger().info("Pick and place sequence complete!")
        
    def go(self):
        plan = self.move_group.go(wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()
        if not plan:
            self.get_logger().warn("Motion planning failed!")
            return False
        return True

def main(args=None):
    rclpy.init(args=args)
    node = PickPlaceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
