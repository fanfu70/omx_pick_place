#!/usr/bin/env python3
"""
Pick & Place Orchestrator (MoveIt 2):
Listens for object pose, plans and executes pick/place trajectory using MoveIt 2 action interface.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped, Pose, Point
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    PositionConstraint,
    OrientationConstraint,
    BoundingVolume,
    MoveItErrorCodes,
)
from control_msgs.action import GripperCommand
import tf_transformations
import math


class PickPlaceNode(Node):
    def __init__(self):
        super().__init__('pick_place_node')
        
        # Configuration
        self.declare_parameter('approach_dist', 0.15)
        self.declare_parameter('lift_dist', 0.20)
        self.declare_parameter('grasp_depth', 0.02)
        self.declare_parameter('place_x', -0.3)
        self.declare_parameter('place_y', 0.0)
        self.declare_parameter('place_z', 0.20)
        self.declare_parameter('arm_group_name', 'arm')
        self.declare_parameter('gripper_action_name', '/gripper_controller/gripper_cmd')
        self.declare_parameter('planning_frame', 'base_link')
        self.declare_parameter('ee_link_name', 'end_effector_link')
        self.declare_parameter('max_planning_time', 5.0)
        self.declare_parameter('planning_attempts', 5)
        self.declare_parameter('retreat_dist', 0.1)
        self.declare_parameter('approach_offset', 0.03)

        self.approach_dist = self.get_parameter('approach_dist').value
        self.lift_dist = self.get_parameter('lift_dist').value
        self.grasp_depth = self.get_parameter('grasp_depth').value
        self.arm_group = self.get_parameter('arm_group_name').value
        self.planning_frame = self.get_parameter('planning_frame').value
        self.ee_link = self.get_parameter('ee_link_name').value
        self.max_planning_time = self.get_parameter('max_planning_time').value
        self.planning_attempts = self.get_parameter('planning_attempts').value
        self.retreat_dist = self.get_parameter('retreat_dist').value
        self.approach_offset = self.get_parameter('approach_offset').value

        # Action clients
        self._moveit_action_client = ActionClient(self, MoveGroup, 'move_action')
        gripper_action = self.get_parameter('gripper_action_name').value
        self._gripper_action_client = ActionClient(self, GripperCommand, gripper_action)

        # Wait for action servers
        self.get_logger().info("Waiting for MoveIt action server...")
        if not self._moveit_action_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("MoveIt action server not available!")
        else:
            self.get_logger().info("MoveIt action server connected.")

        self.get_logger().info("Waiting for Gripper action server...")
        if not self._gripper_action_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("Gripper action server not available!")
        else:
            self.get_logger().info("Gripper action server connected.")

        # State tracking
        self.is_grasping = False
        self.busy = False

        # Subscribers
        self.pose_sub = self.create_subscription(
            PoseStamped, '/object_pose', self.pose_callback, 10
        )

        self.get_logger().info("Pick-Place Orchestrator started. Waiting for object pose...")

    def make_pose(self, position: tuple, rpy: tuple = (0, 0, 0)) -> Pose:
        """Create a geometry_msgs Pose from position and roll-pitch-yaw."""
        pose = Pose()
        pose.position.x = position[0]
        pose.position.y = position[1]
        pose.position.z = position[2]
        q = tf_transformations.quaternion_from_euler(*rpy)
        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]
        return pose

    def is_pose_in_workspace(self, pos: tuple) -> bool:
        """Check if pose is within reasonable workspace bounds."""
        x, y, z = pos
        # Typical workspace bounds for OpenManipulator-X (adjust as needed)
        if z < 0.0 or z > 0.6:
            self.get_logger().warn(f"Z={z} out of workspace bounds [0.0, 0.6]")
            return False
        if math.sqrt(x**2 + y**2) > 0.5:
            self.get_logger().warn(f"XY distance={math.sqrt(x**2 + y**2):.3f} exceeds reach")
            return False
        return True

    def build_moveit_goal(self, pose: Pose) -> MoveGroup.Goal:
        """Build a MoveGroup action goal for a target pose."""
        goal = MoveGroup.Goal()
        goal.request.group_name = self.arm_group
        goal.request.planning_time = self.max_planning_time
        goal.request.num_planning_attempts = self.planning_attempts
        goal.request.allowed_planning_time = self.max_planning_time

        constraint = Constraints()

        # Position constraint
        pos_constraint = PositionConstraint()
        pos_constraint.header.frame_id = self.planning_frame
        pos_constraint.header.stamp = self.get_clock().now().to_msg()
        pos_constraint.link_name = self.ee_link
        pos_constraint.target_point_offset.x = 0.0
        pos_constraint.target_point_offset.y = 0.0
        pos_constraint.target_point_offset.z = 0.0
        pos_constraint.constraint_radius = 0.01  # 1cm tolerance

        box = BoundingVolume()
        box.poses.append(pose)
        box.dimensions.append(Point(x=0.02, y=0.02, z=0.02))
        pos_constraint.bounding_volumes.append(box)
        pos_constraint.weight = 1.0
        constraint.position_constraints.append(pos_constraint)

        # Orientation constraint (loose)
        orient_constraint = OrientationConstraint()
        orient_constraint.header.frame_id = self.planning_frame
        orient_constraint.header.stamp = self.get_clock().now().to_msg()
        orient_constraint.link_name = self.ee_link
        orient_constraint.orientation = pose.orientation
        orient_constraint.absolute_x_axis_tolerance = 0.5
        orient_constraint.absolute_y_axis_tolerance = 0.5
        orient_constraint.absolute_z_axis_tolerance = 0.5
        orient_constraint.weight = 0.5
        constraint.orientation_constraints.append(orient_constraint)

        goal.request.constraints = [constraint]
        return goal

    def move_to_pose(self, pose: Pose) -> bool:
        """Plan and execute motion to a target pose using MoveIt 2 action (blocking)."""
        goal = self.build_moveit_goal(pose)

        future = self._moveit_action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

        goal_handle = future.result()
        if goal_handle is None:
            self.get_logger().error("MoveIt goal rejected")
            return False

        execute_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, execute_future, timeout_sec=self.max_planning_time + 15.0)

        result = execute_future.result()
        if result is None:
            self.get_logger().error("MoveIt action failed or timed out")
            return False

        if result.result.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(f"MoveIt planning failed with error code {result.result.error_code.val}")
            return False

        self.get_logger().info("MoveIt motion executed successfully")
        return True

    def control_gripper(self, position: float, max_effort: float = 50.0) -> bool:
        """
        Control the gripper using GripperCommand action (blocking).

        Args:
            position: position command (0.0=open, 1.0=closed typically)
            max_effort: effort command for the gripper
        """
        goal = GripperCommand.Goal()
        goal.command.position = position
        goal.command.max_effort = max_effort

        future = self._gripper_action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

        goal_handle = future.result()
        if goal_handle is None:
            self.get_logger().error("Gripper goal rejected")
            return False

        execute_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, execute_future, timeout_sec=5.0)

        result = execute_future.result()
        if result is None:
            self.get_logger().error("Gripper action failed")
            return False

        self.get_logger().info(f"Gripper set to position {position}")
        return True

    def pose_callback(self, msg: PoseStamped):
        """Callback when object pose is detected."""
        if self.busy:
            self.get_logger().warn("Already executing pick-and-place, ignoring new pose")
            return

        pos = msg.pose.position
        self.get_logger().info(f"Object detected at: x={pos.x:.3f}, y={pos.y:.3f}, z={pos.z:.3f}")

        # Validate pose
        if not self.is_pose_in_workspace((pos.x, pos.y, pos.z)):
            self.get_logger().warn("Object pose is outside workspace, skipping")
            return

        if pos.z <= 0.01:
            self.get_logger().warn("Object Z too low, possibly invalid depth reading")
            return

        # Execute
        self.busy = True
        self.execute_pick_and_place(pos.x, pos.y, pos.z)

    def execute_pick_and_place(self, obj_x: float, obj_y: float, obj_z: float):
        """Execute the full pick-and-place sequence."""
        place_x = self.get_parameter('place_x').value
        place_y = self.get_parameter('place_y').value
        place_z = self.get_parameter('place_z').value

        try:
            # 1. Pre-grasp (above object)
            self.get_logger().info("Step 1: Moving to pre-grasp position...")
            pre_grasp = self.make_pose((obj_x, obj_y, obj_z + self.approach_dist))
            if not self.move_to_pose(pre_grasp):
                self.get_logger().error("Failed to move to pre-grasp position")
                return

            # 2. Open gripper
            self.get_logger().info("Step 2: Opening gripper...")
            self.control_gripper(0.0)
            self.is_grasping = False

            # 3. Approach (closer to object)
            self.get_logger().info("Step 3: Approaching object...")
            approach = self.make_pose((obj_x, obj_y, obj_z + self.approach_offset))
            if not self.move_to_pose(approach):
                self.get_logger().error("Failed to approach object")
                return

            # 4. Grasp (slightly into object)
            self.get_logger().info("Step 4: Moving to grasp position...")
            grasp = self.make_pose((obj_x, obj_y, obj_z - self.grasp_depth))
            if not self.move_to_pose(grasp):
                self.get_logger().error("Failed to move to grasp position")
                return

            # 5. Close gripper
            self.get_logger().info("Step 5: Closing gripper...")
            self.control_gripper(1.0)
            self.is_grasping = True

            # 6. Lift
            self.get_logger().info("Step 6: Lifting object...")
            lift = self.make_pose((obj_x, obj_y, obj_z + self.lift_dist))
            if not self.move_to_pose(lift):
                self.get_logger().error("Failed to lift object")
                return

            # 7. Move to place position
            self.get_logger().info("Step 7: Moving to place position...")
            place = self.make_pose((place_x, place_y, place_z))
            if not self.move_to_pose(place):
                self.get_logger().error("Failed to move to place position")
                return

            # 8. Open gripper to release
            self.get_logger().info("Step 8: Releasing object...")
            self.control_gripper(0.0)
            self.is_grasping = False

            # 9. Retreat
            self.get_logger().info("Step 9: Retreating...")
            retreat = self.make_pose((place_x, place_y, place_z + self.retreat_dist))
            if not self.move_to_pose(retreat):
                self.get_logger().error("Failed to retreat")
                return

            self.get_logger().info("Pick and place sequence complete!")

        except Exception as e:
            self.get_logger().error(f"Exception during pick-and-place: {e}")
        finally:
            # Safety: ensure gripper is open if something went wrong while grasping
            if self.is_grasping:
                self.get_logger().warn("Error occurred while grasping — opening gripper for safety.")
                self.control_gripper(0.0)
            self.is_grasping = False
            self.busy = False


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
