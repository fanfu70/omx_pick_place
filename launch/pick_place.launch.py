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
                'enable_pointcloud': False # We compute 3D manually for precision
            }]
        ),
        
        # --- Calibration Node ---
        Node(
            package='omx_pick_place',
            executable='calibration_node',
            name='calibration',
            output='screen',
            parameters=[{
                'aruco_dict': 4,
                'aruco_marker_id': 0,
                'marker_size_m': 0.05, # Adjust this to your printed marker size
                'marker_world_x': 0.15, # Ruler measurement from base_link
                'marker_world_y': 0.0,
                'marker_world_z': 0.15
            }]
        ),
        
        # --- Color Detector ---
        Node(
            package='omx_pick_place',
            executable='color_detector',
            name='color_detector',
            output='screen',
            parameters=[{
                'target_color': 'red' # Change to 'blue', 'green', 'yellow'
            }]
        ),
        
        # --- Depth Localizer ---
        Node(
            package='omx_pick_place',
            executable='depth_localizer',
            name='depth_localizer',
            output='screen'
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
                'place_x': -0.3, # Configurable drop-off location
                'place_y': 0.0,
                'place_z': 0.20
            }]
        ),
        
        # --- RViz for Visualization ---
        # Note: You need to manually add /camera/color/image_raw and /object_pose in RViz
        # Node(
        #     package='rviz2',
        #     executable='rviz2',
        #     name='rviz2',
        #     arguments=['-d', '/path/to/your/pick_place.rviz'],
        #     output='screen'
        # ),
    ])
