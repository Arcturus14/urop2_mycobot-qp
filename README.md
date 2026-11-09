# myCobot 280 JN5 closed-loop identification

This workspace separates time-sensitive robot experiments on the Jetson from
model fitting and validation on the laptop.

## Layout

```text
experiments/              # Run on the Jetson (or with --backend mock locally)
  measure_sampling.py
  measure_noise.py
  run_step_test.py
analysis/                 # Run on the laptop
  plot_response.py
  analyze_noise.py
  summarize_acquisition.py
  fit_second_order.py
  validate_model.py
mycobot_id/               # Shared acquisition, CSV, robot, and model code
configs/step_test.example.json
data/raw/                 # Raw measurements (not committed)
data/processed/           # Fitted parameters and derived data
reports/figures/          # Plots
```

## Laptop setup (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-analysis.txt
```

Run the complete pipeline without a robot:

```powershell
python -m experiments.run_step_test --backend mock --config configs/step_test.example.json
python -m analysis.plot_response data/raw/mock_step_J1.csv
python -m analysis.fit_second_order data/raw/mock_step_J1.csv --joint 1
python -m analysis.validate_model data/raw/mock_step_J1.csv data/processed/mock_step_J1_fit.json
python -m experiments.measure_noise --backend mock --duration 2 --output data/raw/mock_noise.csv
python -m analysis.analyze_noise data/raw/mock_noise.csv
```

The mock backend is only a software check. Its fitted parameters must not be
reported as myCobot results.

## Connected Jetson workflow (verified 2026-11-04)

The existing SSH alias `mycobot` connects as `er` to the Jetson `nano`.
The device currently runs Python 3.8.10 with pymycobot 3.5.3 and pyserial 3.4.
Acquisition supports Python 3.8+, while laptop analysis uses Python 3.10+.

The connection configuration is in `configs/jetson.json`. No keys or passwords
are copied into this repository. SSH uses the laptop's existing configuration.

```powershell
.\.venv\Scripts\python.exe tools/jetson.py check
.\.venv\Scripts\python.exe tools/jetson.py deploy
.\.venv\Scripts\python.exe tools/jetson.py collect --samples 200 --duration 20 --poll-interval 0.05
.\.venv\Scripts\python.exe tools/jetson.py fetch
```

Small motion experiments (after confirming a clear workspace and stop access):

```powershell
.\.venv\Scripts\python.exe tools/jetson.py identify --confirm-motion I_UNDERSTAND
```

`identify` uses the measured current pose as its reference and commands one joint
at a time at speed 20, within +/-3 degrees. It does not move the arm to an
all-zero pose. It records +/-1.5 and +/-3 degree targets and reference returns,
twice per joint. Motion envelope, other-joint displacement, invalid serial
responses, and end-of-window stationarity are monitored. Stable target error
is retained as a measured result. The stop command is sent on an aborted run;
there is no automatic return motion after an error.

The complete Korean analysis report can be regenerated from downloaded folders:

```powershell
.\.venv\Scripts\python.exe -m analysis.report_identification <download-folder-1> <download-folder-2> --baseline-download <stationary-download> --output reports/identification_20261104
```

This compares delayed second-order, first-order, velocity-limited, and deadband
velocity-limited candidates. The first repeat trains models; the second repeat
evaluates predictions. The first command per joint is excluded from fitting
because the preceding firmware target is not verified. The output includes
REPORT.md, all event metrics, candidate parameters, and validation figures.

`deploy` creates a new release beneath
`/home/er/workspace/urop2-identification/releases/` and an acquisition environment
at `/home/er/workspace/urop2-identification/.venv`. It reuses the vendor-installed
packages via `--system-site-packages`, without upgrading them. The selected
release is recorded locally in `.jetson/state.json`.

`collect` runs on the Jetson, checks that the serial port is unused, and only
calls `get_angles()`. It writes sampling/noise CSVs and environment metadata.
No position, power, release-servo, or homing command is issued. Noise is measured
at the existing pose; any external motion must be identified from the traces.
Command columns are blank because these runs send no position commands.
CSV samples are flushed as they arrive and existing acquisition files are not
overwritten, so partial data survives an interrupted collection.

`fetch` downloads the current release's raw data into a new timestamped folder
under `data/raw/jetson/` and compares local SHA-256 hashes with the Jetson files.
It leaves the source data in place. The download contains `transfer.json` with
the source release and verified hashes. Run collection and fetch sequentially.

Generate a local report using the folder printed by `fetch`:

```powershell
.\.venv\Scripts\python.exe -m analysis.summarize_acquisition data/raw/jetson/<download-folder> --output reports/jetson/<report-folder>
```

The verified JN serial settings are `/dev/ttyTHS1`, baud `1000000`, matching
[Elephant Robotics' Jetson Nano documentation](https://docs.elephantrobotics.com/docs/mycobot-m5-en/12-ApplicationBaseROS/12.1-ROS1/12.1.4-rivzIntroductionAndUse/myCobot-280.html).
These replace the generic Pi/M5 examples previously shown here.

## Manual Jetson setup

Copy or clone this workspace on the Jetson, create a virtual environment, and
install only the lightweight robot-side requirement file:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-jetson.txt
```

Check the serial device and permissions before moving the robot. A typical
device for this JN is `/dev/ttyTHS1`, but it must be verified on the
actual JN5.

```bash
python -m experiments.measure_sampling --backend real --port /dev/ttyTHS1 --baud 1000000
python -m experiments.measure_noise --backend real --port /dev/ttyTHS1 --baud 1000000 --duration 20
python -m experiments.run_step_test --backend real --port /dev/ttyTHS1 --baud 1000000 \
  --config configs/step_test.example.json --confirm-motion I_UNDERSTAND
```

### Jetson Cartesian IK move (preview first)

`scripts/jetson_cartesian_move.py` reads the current firmware coordinates and
joint angles, preserves the measured RPY, calls pymycobot's
`solve_inv_kinematics(target_coords, current_angles)`, and checks conservative
joint limits plus per-joint displacement. It is a dry run unless the exact
motion confirmation is supplied.

The vendor `pymycobot 4.0.7` currently installed on this Jetson returns this IK
response as integer centidegrees even though ordinary angle reads are converted
to degrees. The script detects that integer response and applies pymycobot's
own `_int2angle()` conversion. Both the raw response and decoded degrees are
printed in the dry-run plan. The observed no-solution response of six repeated
`8241` values is rejected explicitly instead of being misread as six 82.41
degree joints.

```bash
python scripts/jetson_cartesian_move.py --xyz X Y Z
python scripts/jetson_cartesian_move.py --xyz X Y Z \
  --method angles --speed 5 --confirm-motion I_UNDERSTAND
```

After separately validating angle-command runs, compare the firmware Cartesian
command in another run. Mode 0 requests angular interpolation; do not run both
methods automatically back-to-back.

```bash
python scripts/jetson_cartesian_move.py --xyz X Y Z \
  --method coords --coord-mode 0 --speed 5 --confirm-motion I_UNDERSTAND
```

Defaults reject Cartesian moves longer than 30 mm, any joint change over 15
degrees, and IK solutions within 2 degrees of the conservative joint limits.
These endpoint checks do not prove that the swept path is free of self or
environment collisions. Use a clear workspace, keep stop access available, and
begin with a much smaller target displacement.

`--backend real` imports `pymycobot` only at runtime, so laptop analysis does
not require the robot package.

## Safety before the first real run

- Verify joint limits, clear the workspace, and keep the emergency stop ready.
- Begin with one joint, small steps, and low speed.
- Confirm the JN5 serial port and `pymycobot` constructor on the actual image.
- The script requires `--confirm-motion I_UNDERSTAND` for real step motion.
- Raw CSV files record every sample's monotonic timestamps; do not replace them
  with a presumed fixed sampling period.

## CSV timing convention

- `t_s`: sample midpoint relative to experiment start.
- `read_start_s`, `read_end_s`: bounds of the `get_angles()` call.
- `read_duration_s`: serial/API read duration.
- `command_start_s`, `command_end_s`: bounds of the latest motion command.
- `q_cmd_j*`, `q_actual_j*`: commanded and measured angles for all six joints.

The identified delay is therefore an **effective delay** containing command,
serial, firmware, sensor-read, and physical response effects.

## Robot pose GUI (run on the laptop)

The GUI runs locally on the laptop and overlays the measured pose (blue) and
requested pose (orange) in one Matplotlib 3-D scene.  It uses the official JN
URDF joint convention, validates the conservative official joint-limit
intersection, and checks the complete joint-interpolated path with a fast
capsule self-collision approximation.

Run from PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts\laptop_robot_gui.py --enable-motion
```

The Jetson runs only `scripts/robot_bridge.py`; it does not run a GUI or poll the
robot continuously.  Pressing **현재 읽기 + 경로 검사** opens one SSH session,
reads the current joints once, and waits.  If the laptop path check succeeds,
**이동** approves that same session.  The bridge sends the target and reads the
final joints once after motion.  Editing a target invalidates the pending check.

The current pose configuration enables the adaptive gripper and side-mounted
camera. Both display and collision checking use the same oriented boxes. The
gripper envelope uses the published 112 x 94 x 50 mm full-opening dimensions;
the camera uses a box of 82.965 x 16 x 33.965 mm in its CAD axes, obtained from
the official DAE scene and independently checked against the official STEP
solid. The camera box transform composes the URDF flange-to-camera fixed joint,
the collision mesh origin, and the actual mesh bounding-box center. Camera
mounting uses the official standard orientation, supported qualitatively by the
user's photos; the gripper motor offset remains approximate. Adjust
`camera_offset_m`, `camera_rpy_deg`, `gripper_center_m`, and `gripper_rpy_deg`
in `configs/robot_pose_gui.json` if the mounting changes. A 5 mm software margin
is added when checking a tool box against an arm capsule. The common flange
mount, camera/L5 housing interface, and fixed tool/tool overlap are excluded.
Flexible cables and grasped objects are excluded. Restart the laptop GUI to
load changes; the Jetson bridge does not need redeployment for tool geometry.

Official source assets and their hashes are saved in `assets/official/`.
Reproduce the mesh/unit/instance transform analysis with:
`python tools/inspect_camera_model.py --report`. Optional development package
`cadquery-ocp-novtk` additionally checks the STEP solid bounds. The GUI itself
only uses NumPy and does not need a CAD library.

### Gripper control

The laptop GUI provides Open (100), Close (0), a custom opening 0..100, and
gripper speed 1..100 (default 20). These use the installed pymycobot adaptive
gripper API, not servo-zero calibration. Manual gripper actions require an idle
arm and no pending path-check transaction. Cancel a pending check with Stop.
The optional "목표 도착 후 잡기" checkbox uses a separate grasp opening value;
the bridge closes only after the final joint sample is within 2 degrees of the
requested target. The default is unchecked. A stopped gripper is not proof of
successful object capture; there is no object/force feedback in this workflow.
Collision checking continues to use the full-open gripper envelope.

After camera calibration supplies the desired J1..J6 angles, enter them in the
GUI, open the gripper, enable grasp-after-arrival if desired, check the arm path,
and press Move. Camera Cartesian coordinates cannot be entered as joint angles.
Gripper controls require the updated `scripts/robot_bridge.py` deployment.

### Before/after coordinate logs

The "좌표값" tab shows coordinates in separate before/after columns, with units
and two decimal places; a new before sample clears the previous after column.
PowerShell output and the JSONL file retain timestamped current/final joint
angles, firmware coordinates `[X,Y,Z,Rx,Ry,Rz]` in mm/degrees, and independently
calculated URDF flange XYZ in mm. The bridge queries `get_coords()` once during
the current-pose check and once after arm motion, with no continuous coordinate
polling. Missing coordinate responses are logged as null with an error string.
Firmware coordinates use the controller's configured reference/tool settings;
the model flange is not the gripper fingertip or camera optical center.
Each GUI run appends JSON lines to `reports/gui/pose_<timestamp>.jsonl`.

### Table-contact Z reference

`floor_calibration` in the pose config applies a provisional -3.907428 mm Z
translation to model coordinate displays and the plot. Firmware coordinates and
joint commands are unchanged. A third coordinate group estimates the midpoint
of the completely closed fingertips; it is explicitly conditional on closure,
not a live opening-dependent TCP. The point was derived from official finger
meshes, the JN gripper mounting chain and the official slider mapping
API opening 0 -> controller -0.74 rad. Source contact, model inspection, and the
reproducible offline script are under `data/calibration`, `assets/official`, and
`tools/inspect_gripper_contact.py`. The calibration sets the recorded contact
to Z=0 by construction; additional contacts are needed to independently assess
accuracy. It does not calibrate XY, table tilt, joint zero errors, or firmware
reference settings. The drawn table plane is a reference, not an added collision
object; collision checking still covers self-collision only.

### EE coordinate input

The laptop GUI's "EE 좌표" tab accepts table-referenced X/Y/Z in millimetres and
yaw in degrees. The target is the model's fully closed fingertip midpoint, with
flange +Z pointing down. Yaw rotates the flange +X around table +Z; this is not
necessarily the camera image's object angle. Click "IK 계산 + 미리보기", then
"현재 읽기 + 경로 검사", then "EE 위치로 이동". IK alone sends no robot command.
It finds a joint-limited, self-collision-free end pose; the fresh-current path
check is still required. Editing EE inputs cancels pending checks. The tab
shows the numerical floor workspace bounds from `reports/floor_workspace.json`,
which assume Z=0 and free yaw; they are not a guaranteed reachable rectangle
and do not apply unchanged to other heights or fixed yaw. Below-table targets
are rejected, but complete table/case/environment collision checking is not
provided. SciPy is needed on the laptop only (`requirements-analysis.txt`).
