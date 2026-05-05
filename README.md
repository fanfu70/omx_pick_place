# OpenManipulator-X: Vision-Guided Pick & Place

A **ROS 2** pipeline for vision-guided pick-and-place operations using the [OpenManipulator-X (RM-X52-TNM)](https://emanual.robotis.com/docs/en/platform/openmanipulator_x/overview/) robot arm and an Intel RealSense D435 depth camera. The system detects colored objects in a camera image, localizes them in 3D space, and executes autonomous pick-and-place trajectories with MoveIt.

## System Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│   RealSense │────▶│  Color Detector   │────▶│  Depth Localizer  │────▶│  Pick & Place Node  │
│   D435      │     │  (HSV + contours) │     │  (2D→3D + TF)    │     │  (MoveIt planning)  │
│  RGB + Depth│     └──────────────────┘     └──────────────────┘     └─────────────────────┘
└─────────────┘           │                       │                         │
                          │                       │                         │
                    /detected_object         /object_pose             OpenManipulator-X
                    (Pose2D u,v)             (PoseStamped 3D)           Arm + Gripper
```

### Pipeline Stages

| Stage | Node | Input | Output | Description |
|-------|------|-------|--------|-------------|
| 1 | `calibration_node` | RGB image + ArUco marker | Static TF: `world → camera` | One-shot calibration to establish camera-to-world transform |
| 2 | `color_detector` | `/camera/color/image_raw` | `/detected_object` (Pose2D) | HSV color segmentation → largest contour → pixel centroid |
| 3 | `depth_localizer` | `/detected_object` + depth image | `/object_pose` (PoseStamped) | Pixel centroid + depth → 3D point → transform to world frame |
| 4 | `pick_place_node` | `/object_pose` | — | MoveIt trajectory planning: approach → grasp → lift → move → release |

## Hardware Requirements

- **Robot Arm:** OpenManipulator-X (RM-X52-TNM) with DYNAMIXEL-X motors
- **Camera:** Intel RealSense D435 (or D435i) depth camera
- **Controller:** OpenCR board
- **PC:** Ubuntu 22.04 + ROS 2 Humble
- **Calibration:** Printed ArUco marker (DICT_6X6_250, ID 0)

## Prerequisites

### 1. ROS 2 Humble

```bash
sudo apt install ros-humble-desktop
sudo apt install python3-colcon-common-extensions
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### 2. OpenManipulator-X

Follow the official guide: [OpenMANIPULATOR-X Installation](https://emanual.robotis.com/docs/en/platform/openmanipulator_x/installation/)

```bash
mkdir -p ~/omx_ws/src && cd ~/omx_ws/src
git clone https://github.com/ROBOTIS-GIT/open_manipulator_x.git
cd ~/omx_ws && rosdep install --from-paths src --ignore-src -r -y
colcon build && source install/setup.bash
```

### 3. Intel RealSense ROS 2 Driver

```bash
sudo apt install ros-humble-realsense2-camera
```

### 4. Python Dependencies

```bash
pip3 install opencv-python numpy tf-transformations
```

## Installation

### 1. Clone This Repository

```bash
cd ~ && git clone https://github.com/fengting70/omx_pick_place_ws.git
```

### 2. Set Up the Workspace

Copy the package into your ROS 2 workspace:

```bash
# Using your existing OM-X workspace
cp -r omx_pick_place_ws ~/omx_ws/src/omx_pick_place

# Or create a dedicated workspace
mkdir -p ~/pick_place_ws/src
cp -r omx_pick_place_ws ~/pick_place_ws/src/omx_pick_place
```

### 3. Build

```bash
cd ~/pick_place_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

## Usage

### Step 1: Hardware Setup

1. Mount the RealSense D435 camera on the robot arm (end effector or base, facing the workspace)
2. Connect OpenManipulator-X via USB to the PC
3. Plug in the RealSense camera via USB 3.0

### Step 2: Start the Robot

In one terminal, launch the OpenManipulator-X hardware:

```bash
source ~/omx_ws/install/setup.bash
ros2 launch open_manipulator_x_robot open_manipulator_x_robot.launch.py
```

### Step 3: Calibrate Camera Position

Print an ArUco marker (dictionary `DICT_6X6_250`, ID 0) and tape it to a known location on your workspace. Then run:

```bash
source install/setup.bash
ros2 run omx_pick_place calibration_node \
  --ros-args -p aruco_marker_id:=0 \
  -p marker_size_m:=0.05 \
  -p marker_world_x:=0.15 -p marker_world_y:=0.0 -p marker_world_z:=0.15
```

**Adjust the parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `marker_size_m` | Physical size of the printed ArUco marker (meters) | `0.05` (5 cm) |
| `marker_world_x` | Marker X position relative to robot base | `0.15` |
| `marker_world_y` | Marker Y position relative to robot base | `0.0` |
| `marker_world_z` | Marker height above table (meters) | `0.15` |

The node broadcasts a static TF transform (`world → camera_color_optical_frame`) and then exits. **Run this every time you move the camera.**

### Step 4: Run the Full Pipeline

Launch all nodes at once:

```bash
source install/setup.bash
ros2 launch omx_pick_place pick_place.launch.py
```

Or run with custom parameters:

```bash
ros2 launch omx_pick_place pick_place.launch.py \
  target_color:=blue \
  place_x:=-0.3 place_y:=0.0 place_z:=0.20 \
  approach_dist:=0.15 lift_dist:=0.20
```

### Step 5: Place a Colored Object

Place the target-colored object in the camera's field of view. The system will:

1. Detect the object's color and compute its 2D centroid
2. Read the depth value at that pixel to compute the 3D position
3. Transform the pose to the world frame
4. Execute a pick-and-place trajectory

## Configuration

### Color Detector Parameters

| Parameter | Values | Description |
|-----------|--------|-------------|
| `target_color` | `red`, `green`, `blue`, `yellow` | Object color to detect |

### Pick & Place Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `approach_dist` | `0.15` | Distance above object for pre-grasp pose (meters) |
| `lift_dist` | `0.20` | Lift height after grasping (meters) |
| `grasp_depth` | `0.02` | How deep the gripper descends into the object (meters) |
| `place_x` | `-0.3` | Drop-off X coordinate |
| `place_y` | `0.0` | Drop-off Y coordinate |
| `place_z` | `0.20` | Drop-off height |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `TF Transform failed` | Run the calibration node first; ensure the ArUco marker is visible |
| Object not detected | Check `target_color` parameter; ensure lighting is adequate; verify HSV thresholds in `utils.py` |
| Depth reads 0 or NaN | Check the camera's depth field of view; ensure the object is 0.2–2.0 m from the camera |
| MoveIt planning fails | Check robot arm is powered on; verify URDF matches your configuration; reduce `approach_dist` |
| Gripper doesn't close | Verify `gripper_left_joint` name matches your URDF; check DYNAMIXEL port handler is running |
| RealSense not detected | `sudo modprobe uvcvideo`; ensure USB 3.0 connection; check `ls /dev/video*` |

## Project Structure

```
omx_pick_place/
├── launch/
│   └── pick_place.launch.py    # Launch all pipeline nodes
├── omx_pick_place/
│   ├── __init__.py
│   ├── calibration_node.py     # ArUco-based camera calibration
│   ├── color_detector.py       # HSV color segmentation
│   ├── depth_localizer.py      # 2D centroid → 3D pose
│   ├── pick_place_node.py      # MoveIt pick & place orchestration
│   └── utils.py                # Color thresholds, morphological ops
├── package.xml
├── setup.cfg
└── setup.py
```

## License

Apache-2.0
