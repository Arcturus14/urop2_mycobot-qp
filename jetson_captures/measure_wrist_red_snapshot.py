"""Measure red patches on an already-undistorted, fixed-pose table snapshot.

Uses the calibrated hole convex hull as ROI. Does not access camera or robot.
Robot XY is provisional; block height is ignored. Image must use calibration pose.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--image', required=True)
parser.add_argument('--calibration', required=True)
parser.add_argument('--output', required=True, help='New result prefix; never overwrite')
args = parser.parse_args()
out_json = Path(args.output + '.json')
out_image = Path(args.output + '.jpg')
if out_json.exists() or out_image.exists():
    raise SystemExit('Output already exists; choose a new prefix')
with open(args.calibration) as f:
    calibration = json.load(f)
im = cv2.imread(args.image)
assert im is not None, 'Cannot read image'
assert list(im.shape[1::-1]) == calibration['resolution'], 'Resolution mismatch'
hull = cv2.convexHull(np.float32(calibration['pixel']))
roi = np.zeros(im.shape[:2], np.uint8)
cv2.fillConvexPoly(roi, np.rint(hull).astype(np.int32), 255)
hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, (0, 90, 45), (12, 255, 255)) | cv2.inRange(hsv, (168, 90, 45), (179, 255, 255))
mask &= roi
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
H = np.float64(calibration['H_pixel_to_grid_mm'])
R = np.float64(calibration['H_pixel_to_robot_xy_PROVISIONAL'])
results = []
for contour in sorted(contours, key=cv2.contourArea, reverse=True):
    if cv2.contourArea(contour) < 100:
        continue
    # Reject patches clipped by the calibrated area instead of reporting a partial center.
    boundary_distance = cv2.distanceTransform(roi, cv2.DIST_L2, 3)
    xy = contour.reshape(-1, 2)
    if np.min(boundary_distance[xy[:, 1], xy[:, 0]]) < 2:
        continue
    plane_contour = cv2.perspectiveTransform(contour.astype(np.float64), H)
    moments = cv2.moments(plane_contour.astype(np.float32))
    if abs(moments['m00']) < 1e-9:
        continue
    grid = np.array([moments['m10'] / moments['m00'], moments['m01'] / moments['m00']])
    pixel = cv2.perspectiveTransform(grid.reshape(1, 1, 2), np.linalg.inv(H))[0, 0]
    robot = cv2.perspectiveTransform(pixel.reshape(1, 1, 2), R)[0, 0]
    result = {'pixel_center': pixel.tolist(), 'grid_xy_mm': grid.tolist(),
              'robot_xy_mm_provisional': robot.tolist(), 'red_area_px': cv2.contourArea(contour)}
    results.append(result)
    cv2.drawContours(im, [contour], -1, (0, 255, 0), 2)
    ctr = tuple(np.rint(pixel).astype(int))
    cv2.drawMarker(im, ctr, (0, 255, 255), cv2.MARKER_CROSS, 16, 2)
    cv2.putText(im, 'grid (%.1f, %.1f) mm' % tuple(grid), (ctr[0]-70, ctr[1]-35), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 0), 1)
    cv2.putText(im, 'robot XY~(%.1f, %.1f)' % tuple(robot), (ctr[0]-70, ctr[1]+55), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 255), 1)
record = {'source': args.image, 'calibration': args.calibration,
          'grid_axes': calibration['grid_axes'], 'height_ignored': True,
          'pose_requirement': calibration['capture_robot_coords'],
          'caution': 'Robot XY provisional; independent robot validation pending. Pose must match. No motion authorized by this output.',
          'objects': results}
with out_json.open('x') as f:
    json.dump(record, f, indent=2)
assert cv2.imwrite(str(out_image), im)
print(json.dumps(record))
