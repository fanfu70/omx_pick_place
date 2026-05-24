#!/usr/bin/env python3
"""
Pick & Place Orchestrator (MoveIt 2):
Listens for object pose, plans and executes pick/place trajectory using MoveIt 2 action interface.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped, Pose
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    PositionConstraint,
    OrientationConstraint,
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
        self.declare_parameter('place_x', float('nan'))
        self.declare_parameter('place_y', float('nan'))
        self.declare_parameter('place_z', float('nan'))
        self.declare_parameter('arm_group_name', 'arm')
        self.declare_parameter('gripper_action_name', '/gripper_controller/gripper_cmd')
        self.declare_parameter('planning_frame', 'base_link')
        self.declare_parameter('ee_link_name', 'end_effector_link')
        self.declare_parameter('max_planning_time', 5.0)
        self.declare_parameter('planning_attempts', 5)
        self.declare_parameter('retreat_dist', 0.1)
        self.declare_parameter('approach_offset', 0.03)
        self.declare_parameter('max_steps', 0)

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

        # max_steps: 0 means run all 9 steps
        self.max_steps = self.get_parameter('max_steps').value
        if self.max_steps <= 0 or self.max_steps > 9:
            self.max_steps = 9
            self.get_logger().info('max_steps not set or invalid, defaulting to all 9 steps')
        else:
            self.get_logger().info(f'max_steps set to {self.max_steps}')

        # Action clients
        self._moveit_action_client = ActionClient(self, MoveGroup, 'move_action')
        gripper_action = self.get_parameter('gripper_action_name').value
        self._gripper_action_client = ActionClient(self, GripperCommand, gripper_action)

        # Wait for action servers
        self.get_logger().info("Waiting for MoveIt action server...")
        self._moveit_server_available = self._moveit_action_client.wait_for_server(timeout_sec=10.0)
        if not self._moveit_server_available:
            self.get_logger().error("MoveIt action server not available!")
        else:
            self.get_logger().info("MoveIt action server connected.")

        self.get_logger().info("Waiting for Gripper action server...")
        self._gripper_server_available = self._gripper_action_client.wait_for_server(timeout_sec=10.0)
        if not self._gripper_server_available:
            self.get_logger().error("Gripper action server not available!")
        else:
            self.get_logger().info("Gripper action server connected.")

        # Cache place parameters to avoid re-reading on every callback
        self.place_x = self.get_parameter('place_x').value
        self.place_y = self.get_parameter('place_y').value
        self.place_z = self.get_parameter('place_z').value

        # State tracking
        self.is_grasping = False
        self.busy = False

        # Subscribers
        self.pose_sub = self.create_subscription(
            PoseStamped, '/object_pose', self.pose_callback, 10
        )

        self.get_logger().info("Pick-Place Orchestrator started. Waiting for object pose...")

    def wait_for_future(self, future, timeout_sec: float) -> bool:
        """Wait for a future to complete without spinning the executor.
        
        This is safe to call when the node is already being spun by rclpy.spin().
        Uses a polling approach with spin_once to avoid blocking the executor.
        """
        start = self.get_clock().now()
        while not future.done():
            if (self.get_clock().now() - start).nanoseconds / 1e9 >= timeout_sec:
                self.get_logger().warn(f"Future timeout after {timeout_sec}s")
                return False
            # Yield to the executor so it can process callbacks
            rclpy.spin_once(self, timeout_sec=0.1)
        return True

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

        prim = SolidPrimitive()
        prim.type = SolidPrimitive.BOX
        prim.dimensions = [0.02, 0.02, 0.02]
        pos_constraint.constraint_region.primitives.append(prim)
        pos_constraint.constraint_region.primitive_poses.append(pose)
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

        goal.request.goal_constraints = [constraint]
        return goal

    def move_to_pose(self, pose: Pose) -> bool:
        """Plan and execute motion to a target pose using MoveIt 2 action (blocking)."""
        goal = self.build_moveit_goal(pose)

        self.get_logger().info("Sending MoveIt goal...")
        future = self._moveit_action_client.send_goal_async(goal)
        if not self.wait_for_future(future, 5.0):
            self.get_logger().error("MoveIt goal timeout")
            return False

        try:
            goal_handle = future.result()
        except Exception as e:
            self.get_logger().error(f"MoveIt goal failed with exception: {e}")
            return False
        if goal_handle is None:
            self.get_logger().error("MoveIt goal rejected")
            return False

        self.get_logger().info("MoveIt goal accepted, waiting for execution...")
        execute_future = goal_handle.get_result_async()
        if not self.wait_for_future(execute_future, self.max_planning_time + 15.0):
            self.get_logger().error("MoveIt action timed out")
            return False

        try:
            result = execute_future.result()
        except Exception as e:
            self.get_logger().error(f"MoveIt action failed with exception: {e}")
            return False
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
        if not self.wait_for_future(future, 5.0):
            self.get_logger().error("Gripper goal timeout")
            return False

        try:
            goal_handle = future.result()
        except Exception as e:
            self.get_logger().error(f"Gripper goal failed with exception: {e}")
            return False
        if goal_handle is None:
            self.get_logger().error("Gripper goal rejected")
            return False

        execute_future = goal_handle.get_result_async()
        if not self.wait_for_future(execute_future, 5.0):
            self.get_logger().error("Gripper action timed out")
            return False

        try:
            result = execute_future.result()
        except Exception as e:
            self.get_logger().error(f"Gripper action failed with exception: {e}")
            return False
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
        raw_place_x = self.place_x
        raw_place_y = self.place_y
        raw_place_z = self.place_z

        # If place parameters not provided, default to object pose
        if math.isnan(raw_place_x):
            place_x = obj_x
            self.get_logger().info(f"place_x not provided, defaulting to object pose.x: {place_x:.3f}")
        else:
            place_x = raw_place_x

        if math.isnan(raw_place_y):
            place_y = -obj_y
            self.get_logger().info(f"place_y not provided, defaulting to -object pose.y: {place_y:.3f}")
        else:
            place_y = raw_place_y

        if math.isnan(raw_place_z):
            place_z = obj_z
            self.get_logger().info(f"place_z not provided, defaulting to object pose.z: {place_z:.3f}")
        else:
            place_z = raw_place_z

        try:
            # 1. Pre-grasp (above object)
            if self.max_steps >= 1:
                self.get_logger().info("Step 1: Moving to pre-grasp position...")
                pre_grasp = self.make_pose((obj_x, obj_y, obj_z + self.approach_dist))
                if not self.move_to_pose(pre_grasp):
                    self.get_logger().error("Failed to move to pre-grasp position")
                    return
            else:
                self.get_logger().info("Step 1 skipped (max_steps < 1)")

            # 2. Open gripper
            if self.max_steps >= 2:
                self.get_logger().info("Step 2: Opening gripper...")
                self.control_gripper(0.0)
                self.is_grasping = False
            else:
                self.get_logger().info("Step 2 skipped (max_steps < 2)")

            # 3. Approach (closer to object)
            if self.max_steps >= 3:
                self.get_logger().info("Step 3: Approaching object...")
                approach = self.make_pose((obj_x, obj_y, obj_z + self.approach_offset))
                if not self.move_to_pose(approach):
                    self.get_logger().error("Failed to approach object")
                    return
            else:
                self.get_logger().info("Step 3 skipped (max_steps < 3)")

            # 4. Grasp (slightly into object)
            if self.max_steps >= 4:
                self.get_logger().info("Step 4: Moving to grasp position...")
                grasp = self.make_pose((obj_x, obj_y, obj_z - self.grasp_depth))
                if not self.move_to_pose(grasp):
                    self.get_logger().error("Failed to move to grasp position")
                    return
            else:
                self.get_logger().info("Step 4 skipped (max_steps < 4)")

            # 5. Close gripper
            if self.max_steps >= 5:
                self.get_logger().info("Step 5: Closing gripper...")
                self.control_gripper(1.0)
                self.is_grasping = True
            else:
                self.get_logger().info("Step 5 skipped (max_steps < 5)")

            # 6. Lift
            if self.max_steps >= 6:
                self.get_logger().info("Step 6: Lifting object...")
                lift = self.make_pose((obj_x, obj_y, obj_z + self.lift_dist))
                if not self.move_to_pose(lift):
                    self.get_logger().error("Failed to lift object")
                    return
            else:
                self.get_logger().info("Step 6 skipped (max_steps < 6)")

            # 7. Move to place position
            if self.max_steps >= 7:
                self.get_logger().info("Step 7: Moving to place position...")
                place = self.make_pose((place_x, place_y, place_z))
                if not self.move_to_pose(place):
                    self.get_logger().error("Failed to move to place position")
                    return
            else:
                self.get_logger().info("Step 7 skipped (max_steps < 7)")

            # 8. Open gripper to release
            if self.max_steps >= 8:
                self.get_logger().info("Step 8: Releasing object...")
                self.control_gripper(0.0)
                self.is_grasping = False
            else:
                self.get_logger().info("Step 8 skipped (max_steps < 8)")

            # 9. Retreat
            if self.max_steps >= 9:
                self.get_logger().info("Step 9: Retreating...")
                retreat = self.make_pose((place_x, place_y, place_z + self.retreat_dist))
                if not self.move_to_pose(retreat):
                    self.get_logger().error("Failed to retreat")
                    return
            else:
                self.get_logger().info("Step 9 skipped (max_steps < 9)")

            self.get_logger().info("Pick and place sequence complete!")

        except Exception as e:
            self.get_logger().error(f"Exception during pick-and-place: {e}")
        finally:
            # Safety: ensure gripper is open if something went wrong while grasping
            if self.is_grasping:
                self.get_logger().warn("Error occurred while grasping — opening gripper for safety.")
                try:
                    self.control_gripper(0.0)
                except Exception:
                    self.get_logger().error("Failed to open gripper during safety cleanup")
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
        # Wait for any ongoing operation to complete before shutdown
        if node.busy:
            node.get_logger().warn("Shutting down while busy, waiting for completion...")
            timeout = node.get_clock().now()
            while node.busy:
                rclpy.spin_once(node, timeout_sec=0.1)
                if (node.get_clock().now() - timeout).nanoseconds / 1e9 > 10.0:
                    node.get_logger().error("Timed out waiting for busy operation to complete")
                    break
        
        # Clean up resources
        if hasattr(node, 'pose_sub'):
            node.destroy_subscription(node.pose_sub)
        node.destroy_node()
        rclpy.shutdown()
