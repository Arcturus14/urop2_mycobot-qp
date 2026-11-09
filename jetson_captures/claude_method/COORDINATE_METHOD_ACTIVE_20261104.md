# Active coordinate extraction method: Claude 2026-11-04

User selected the Claude method and reports in `coord_check_20261104/`.
Use `claude_table_method_20261104.json` as the active global grid-to-base XY definition.

## Coordinate convention

S is the grid origin. u=right, w=inward (from the operator standing on the port side), pitch=25mm.
XY = (-46.2,-62.1) + 25 R(-92.40 degrees) (u,w).
XY is in base axes. Output Z=0 is the table-relative declared height, not a sensor reading or base-Z measurement.

Bolts: A=(0,8), B=(4,1), C=(4,8), F=(-7,8), E=(-7,14), D=(0,14).
Use bolt/grid identities, never assume labels from screen clockwise order.

## Image workflow

1. Capture wrist camera at 640x480 and apply `camera_params_wrist.npz` undistortion.
2. Record camera pose and coordinate settings if serial is available. Do not change settings during extraction.
3. Identify table holes and known bolts; map to global S-based u,w. Use at least 5-6 distributed plane hole centers; screw heads are raised and only establish identities.
4. Fit a pixel-to-grid homography; validate held-out holes and check correct axis signs and integer steps.
5. Find the red-region plane center, then transform its global u,w with the active formula.
6. Refit/verify grid correspondence whenever wrist pose or image landmarks change. Never reuse a fixed-camera homography blindly.
7. Report [X,Y,0] mm as a coordinate estimate with source image and uncertainty context. Height is ignored by user request.

The supplied folder contains reports, NOT the referenced original Claude blob/lattice detector and SciPy fitting scripts. Do not claim to have executed those originals. The conversion utility reproduces the published formula and examples; camera-specific hole detection remains separately verified work.

## Controller state and previous outputs

The model TCP [-0.6,15.7,104.3,0,0,0] and end_type=1 were set earlier. Reports reject this model as a physical TCP calibration. Do not use it as the anchor or apply it again to extracted XY. This workflow does not reset the controller or approve motion. Any later controller change must be explicit and documented.

Previous F=[143.1,93.6,0] and red 13:41=[205.5,37.3,0] are superseded.
Claude reference for the saved 13:41 target is [224.5,49.7,0], u,w=(-4.92,10.63).

The reports' fit RMS/uncertainty are source claims, not newly established end-to-end accuracy. D/E are extrapolation points. A 280mm radial test is only a rough screen, not a complete IK/collision/reachability check.

## Commands

```sh
python3 claude_table_coordinates.py --check
python3 claude_table_coordinates.py --u -4.92 --w 10.63
python3 claude_table_coordinates.py --bolt F
```
