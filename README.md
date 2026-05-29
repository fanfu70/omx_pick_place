# omx_pick_place

Vision-guided pick-and-place for [OpenManipulator-X](https://emanual.robotis.com/docs/en/platform/openmanipulator_x/overview/) using a RealSense D435 camera and MoveIt 2.

[![ROS2](https://img.shields.io/badge/ROS2-Foxy%20Fitzroy-blue)](https://docs.ros.org/en/foxy/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)

## Overview

This package implements a complete vision-guided pick-and-place system for the OpenManipulator-X robotic arm. It detects colored objects using RGB-D vision, localizes them in 3D world coordinates, and commands the robot arm to perform precise pick-and-place operations using MoveIt 2 planning.

### System Architecture

The system consists of four main components:

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Calibration   │     │   Color          │     │   Depth         │     │   Pick & Place  │
│   Node          │────▶│   Detector       │────▶│   Localizer     │────▶│   Orchestrator  │
└─────────────────┘     └──────────────────┘     └─────────────────┘     └─────────────────┘
        │                      │                       │                        │
        ▼                      ▼                       ▼                        ▼
  ArUco Markers        RGB Camera           Depth Camera           MoveIt 2 + Gripper
  Transform          Color Detection       3D Localization         Trajectory Plan
```

### Features

- **Color-based object detection**: Detects red, green, blue, and yellow objects
- **3D localization**: Converts 2D image coordinates to 3D world coordinates using depth data
- **Calibration**: Automated camera-to-world calibration using dual ArUco markers
- **Motion planning**: Uses MoveIt 2 for collision-free trajectory planning
- **Gripper control**: Integrated gripper control for pick-and-place operations
- **Visualization**: Real-time detection visualization and debugging tools

## Requirements

### Hardware

- OpenManipulator-X robot arm
- RealSense D435 depth camera
- Computer with ROS 2 installation (tested with Foxy Fitzroy)
- Internet connection for dependencies

### Software Dependencies

- ROS 2 (Foxy or later)
- OpenCV (`python3-opencv`)
- NumPy (`numpy`)
- tf_transformations (`tf_transformations`)
- matplotlib (`matplotlib`)
- realsense2_camera
- moveit_ros_move_group
- moveit_msgs
- control_msgs
- joint_trajectory_controller

## Installation

### Clone the Package

```bash
cd ~/ros2_ws/src
git clone <your-repository-url>
cd ..
colcon build --packages-select omx_pick_place
source install/setup.bash
```

### Install Dependencies

```bash
rosdep install --from-paths src --ignore-src -r -y
```

## Quick Start

### 1. Calibration (One-time Setup)

Before using the pick-and-place system, you must calibrate the camera to world transformation using ArUco markers.

#### Setup

Place two ArUco markers on your work surface:

- **Marker 1**: Positioned at approximately (0.15, 0.0, 0.15) meters in world coordinates
- **Marker 2**: Positioned at approximately (0.20, 0.0, 0.15) meters in world coordinates

Both markers should be face-up (Z-axis aligned with world Z-axis).

#### Run Calibration

```bash
# Terminal 1: Start the calibration node
ros2 run omx_pick_place calibration_node \
  --ros-args \
    -p aruco_dict:=4 \
    -p aruco_marker_id_1:=0 \
    -p aruco_marker_id_2:=1 \
    -p marker_size_m:=0.05 \
    -p marker1_world_x:=0.15 \
    -p marker1_world_y:=0.0 \
    -p marker1_world_z:=0.15 \
    -p marker1_world_roll:=0.0 \
    -p marker1_world_pitch:=0.0 \
    -p marker1_world_yaw:=0.0 \
    -p marker2_world_x:=0.20 \
    -p marker2_world_y:=0.0 \
    -p marker2_world_z:=0.15 \
    -p marker2_world_roll:=0.0 \
    -p marker2_world_pitch:=0.0 \
    -p marker2_world_yaw:=0.0
```

#### Calibration Process

1. The node subscribes to the camera stream and waits for both ArUco markers
2. Once both markers are detected, it computes the camera-to-world transformation
3. The calibration result is broadcast as a static TF transform
4. Visualizations are saved to `~/calibration_results/` (if debug mode is enabled)

#### Calibration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `aruco_dict` | 4 | ArUco dictionary ID (4 = DICT_6X6_250) |
| `aruco_marker_id_1` | 0 | ID of first marker |
| `aruco_marker_id_2` | 1 | ID of second marker |
| `marker_size_m` | 0.05 | Size of markers in meters |
| `marker*_world_x/y/z` | - | World position of marker center |
| `marker*_world_roll/pitch/yaw` | 0.0 | Orientation of marker in world frame |

### 2. Run Pick-and-Place Pipeline

After calibration, run the complete pipeline:

```bash
# Terminal 2: Launch the system
ros2 launch omx_pick_place pick_place.launch.py
```

### 3. Execute Pick-and-Place

Once the system is running and an object is detected:

```bash
# Terminal 3: Trigger pick-and-place (object must be detected first)
# The system automatically starts when an object is detected
```

## Configuration

### Color Detection Parameters

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| `target_color` | 'red' | red, green, blue, yellow | Color to detect |
| `min_contour_area` | 100 | integer | Minimum object area in pixels |
| `pub_rate_hz` | 10.0 | float | Maximum detection publication rate |

### Pick-and-Place Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `approach_dist` | 0.15 | Distance above object to approach (meters) |
| `lift_dist` | 0.20 | Distance to lift object after grasp (meters) |
| `grasp_depth` | 0.02 | Depth into object for grasp (meters) |
| `place_x` | NaN | X position for placing (uses object X if NaN) |
| `place_y` | NaN | Y position for placing (uses -object Y if NaN) |
| `place_z` | NaN | Z position for placing (uses object Z if NaN) |
| `max_steps` | 0 | Maximum steps to execute (0 = all steps) |
| `home_x/y/z` | 0.163, 0.0, 0.20 | Home position coordinates |

### Depth Localization Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `depth_window_size` | 5 | Size of median filter window |
| `max_depth` | 3.0 | Maximum valid depth in meters |

## Pipeline Details

### 1. Calibration Node (`calibration_node.py`)

The calibration node performs the following steps:

1. Subscribes to camera color stream and camera info
2. Detects two ArUco markers in the image
3. Estimates pose of each marker using `solvePnP`
4. Computes camera pose in world frame using both markers
5. Averages the two computed poses for improved accuracy
6. Broadcasts the transformation `world → camera_link`

**Calibration Math**:
- ArUco provides marker pose in camera frame: `T_cam_marker`
- Invert to get camera pose in marker frame: `T_marker_cam = T_cam_marker^(-1)`
- Transform marker to world using input parameters: `T_world_marker`
- Compose: `T_world_cam = T_world_marker * T_marker_cam`

### 2. Color Detector Node (`color_detector.py`)

The color detector performs:

1. Converts BGR image to HSV color space
2. Applies color thresholding to create binary mask
3. Performs morphological operations to clean the mask
4. Finds the largest contour and computes its centroid
5. Publishes 2D pixel coordinates and annotated images

**Supported Colors**:
- **Red**: Handles HSV wraparound (0-15 and 160-180)
- **Green**: 40-80 hue range
- **Blue**: 100-140 hue range
- **Yellow**: 20-35 hue range

### 3. Depth Localizer Node (`depth_localizer.py`)

The depth localizer:

1. Receives 2D centroid from color detector
2. Samples depth values in a 5x5 window around the centroid
3. Computes median depth to reduce noise
4. Converts 2D-3D using camera intrinsics:
   - `x = (u - cx) * z / fx`
   - `y = (v - cy) * z / fy`
   - `z = depth_value`
5. Transforms point from camera frame to world frame using TF2

### 4. Pick & Place Orchestrator (`pick_place_node.py`)

The orchestrator implements a 10-step state machine:

| Step | Action | Description |
|------|--------|-------------|
| 1 | Pre-grasp | Move above object at approach height |
| 2 | Open gripper | Prepare for grasp |
| 3 | Approach | Move closer to object |
| 4 | Grasp | Move into object for grip |
| 5 | Close gripper | Secure the object |
| 6 | Lift | Raise object off surface |
| 7 | Move to place | Transport to drop location |
| 8 | Release | Open gripper to drop object |
| 9 | Go home | Return to safe home position |
| 10 | Close gripper | Reset gripper for next cycle |

**Safety Features**:
- Workspace bounds validation
- Pose feasibility checking
- Error recovery with safe return to home
- Automatic gripper release on error

## TF Frames

The system uses the following coordinate frames:

```
world → camera_link → camera_color_optical_frame
```

- **world**: Calibration-defined reference frame
- **camera_link**: Physical camera mount
- **camera_color_optical_frame**: Camera sensor coordinate system (ROS convention: Z-forward, X-right, Y-down)

## Debug Visualization

The calibration node supports debug visualization:

```bash
# Enable debug visualization
ros2 run omx_pick_place calibration_node \
  --ros-args -p debug_visualization:=True \
             -p output_dir:=~/calibration_results
```

This generates PNG files in the output directory:
- `calibration_poses.png`: 3D visualization of all frames
- `world_to_marker*.png`: Marker position verification
- `camera_with_markers_pose.png`: Overlaid marker detection
- And more...

## Troubleshooting

### Calibration Issues

**Problem**: ArUco markers not detected
- Ensure good lighting conditions
- Check that markers are clearly visible in camera view
- Verify marker IDs match parameters
- Adjust marker positions if needed

**Problem**: Poor calibration accuracy
- Use larger marker separation (increase marker distance)
- Ensure markers are planar and flat
- Increase marker size for better detection
- Verify marker position parameters match actual placement

### Detection Issues

**Problem**: Object not detected
- Adjust color threshold parameters
- Ensure object has high color contrast
- Check minimum contour area setting
- Verify object is within camera FOV

**Problem**: False detections
- Increase minimum contour area
- Adjust color thresholds for stricter matching
- Add morphological operations if needed

### Motion Planning Issues

**Problem**: Planning fails
- Verify TF frames are correctly set up
- Check that target pose is within workspace
- Increase `max_planning_time` and `planning_attempts`
- Verify collision mesh configuration

**Problem**: Gripper not responding
- Verify gripper action topic name
- Check gripper controller is running
- Verify gripper is properly connected

## API Documentation

### Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/camera/camera/color/image_raw` | sensor_msgs/Image | RGB camera input |
| `/camera/camera/aligned_depth_to_color/image_raw` | sensor_msgs/Image | Depth image |
| `/camera/camera/color/camera_info` | sensor_msgs/CameraInfo | Camera intrinsics |
| `/detected_object` | geometry_msgs/Pose2D | 2D object centroid |
| `/object_pose` | geometry_msgs/PoseStamped | 3D object pose in world |
| `/detection_marker` | visualization_msgs/Marker | RViz visualization |

### Action Servers

| Server | Type | Description |
|--------|------|-------------|
| `/move_action` | MoveGroup | MoveIt 2 motion planning |
| `/gripper_controller/gripper_cmd` | GripperCommand | Gripper control |

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Authors

- OpenManipulator-X development team
- Vision-guided pick-and-place implementation

## Acknowledgments

- [OpenManipulator-X](https://emanual.robotis.com/docs/en/platform/openmanipulator_x/overview/) by ROBOTIS
- [MoveIt 2](https://moveit.ros.org/) for motion planning
- [RealSense](https://github.com/IntelRealSense/realsense-ros) for depth camera support

## Changelog

### v0.1.0
- Initial release
- Color-based object detection
- 3D pose estimation using depth
- Pick-and-place with MoveIt 2
- ArUco-based camera calibration
- Debug visualization tools
- Multi-color support (red, green, blue, yellow)

## Future Enhancements

- Multi-object detection and sorting
- Object classification and orientation estimation
- Adaptive calibration for dynamic environments
- Trajectory optimization and time-parameterization
- Collision avoidance with environment models
- Force control for robust grasping
- Real-time object tracking
- Deep learning-based detection for non-colored objects
- User-friendly GUI for operation and monitoring
- Multi-robot coordination support

