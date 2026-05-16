from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        
        # --- RealSense Camera ---
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name='realsense',
            output='screen',
            parameters=[{
                'enable_color': True,
                'enable_depth': True,
                'color_width': 640,
                'color_height': 480,
                'depth_width': 640,
                'depth_height': 480,
                'enable_pointcloud': False  # We compute 3D manually for precision
            }]
        ),
        
        # --- Calibration Node ---
        # NOTE: Calibration is a one-shot operation. Run it separately before the main pipeline.
        # To calibrate, run:
        #   ros2 run omx_pick_place calibration_node --ros-args -p marker_size_m:=0.05 \
        #     -p marker_world_x:=0.15 -p marker_world_y:=0.0 -p marker_world_z:=0.15
        # The calibration node broadcasts a static TF and then exits.
        # Node(
        #     package='omx_pick_place',
        #     executable='calibration_node',
        #     name='calibration',
        #     output='screen',
        #     parameters=[{
        #         'aruco_dict': 4,
        #         'aruco_marker_id': 0,
        #         'marker_size_m': 0.05,
        #         'marker_world_x': 0.15,
        #         'marker_world_y': 0.0,
        #         'marker_world_z': 0.15,
        #         'marker_world_roll': 0.0,
        #         'marker_world_pitch': 1.5708,
        #         'marker_world_yaw': 0.0
        #     }]
        # ),
        
        # --- Color Detector ---
        Node(
            package='omx_pick_place',
            executable='color_detector',
            name='color_detector',
            output='screen',
            parameters=[{
                'target_color': 'red',  # Change to 'blue', 'green', 'yellow'
                'min_contour_area': 100,
                'pub_rate_hz': 10.0
            }]
        ),
        
        # --- Depth Localizer ---
        Node(
            package='omx_pick_place',
            executable='depth_localizer',
            name='depth_localizer',
            output='screen',
            parameters=[{
                'depth_window_size': 5,
                'max_depth': 3.0
            }]
        ),
        
        # --- Pick & Place Orchestrator ---
        Node(
            package='omx_pick_place',
            executable='pick_place_node',
            name='pick_place',
            output='screen',
            parameters=[{
                'approach_dist': 0.15,
                'lift_dist': 0.20,
                'grasp_depth': 0.02,
                'place_x': -0.3,
                'place_y': 0.0,
                'place_z': 0.20,
                'arm_group_name': 'arm',
                'gripper_action_name': '/gripper_controller/gripper_cmd',
                'planning_frame': 'base_link',
                'ee_link_name': 'end_effector_link',
                'max_planning_time': 5.0,
                'planning_attempts': 5,
                'retreat_dist': 0.1,
                'approach_offset': 0.03
            }]
        ),
        
        # --- RViz for Visualization ---
        # Note: You need to manually add /camera/color/image_raw and /object_pose in RViz
        # Node(
        #     package='rviz2',
        #     executable='rviz2',
        #     name='rviz2',
        #     arguments=['-d', '/path/to/your/pick_place.rviz'],
        #     output='screen',
        # ),
    ])
