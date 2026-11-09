"""Provisional, fixed-snapshot grid fit. No robot or camera access.

Grid origin A; +r is A->C, +in is B->C. Seed identities require human review.
Run in /home/er/camera_vision with existing OpenCV/NumPy.
"""
import json
import cv2
import numpy as np

SOURCE = 'wrist_table_wide_20261104_123139.jpg'
PREFIX = 'wrist_grid_20261104_123139'
im = cv2.imread(SOURCE)
assert im is not None
# Approximate screw centers are only for assigning integer grid indices.
# They are NOT used as plane calibration correspondences (screw heads are raised).
seed_pix = np.float32([[327, 50], [459, 156], [276, 376]])
seed_grid = np.float32([[0, 0], [4, 0], [4, -7]])
initial = cv2.getAffineTransform(seed_pix, seed_grid)
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 65, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
points = []
for contour in contours:
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    if not (12 < area < 180 and 0.55 < w / h < 1.7 and x > 220 and 15 < y < 440):
        continue
    m = cv2.moments(contour)
    p = np.array([m['m10'] / m['m00'], m['m01'] / m['m00']])
    grid = initial @ np.r_[p, 1]
    index = np.rint(grid)
    if np.linalg.norm(grid - index) < 0.24 and -1 <= index[0] <= 5 and -7 <= index[1] <= 1:
        points.append((p, index))
assert len(points) >= 8, 'Insufficient unambiguous hole candidates'
indices = [tuple(q) for _, q in points]
assert len(indices) == len(set(indices)), 'Duplicate grid assignments'
pix = np.float64([p for p, _ in points])
mm = np.float64([q * 25 for _, q in points])
H, _ = cv2.findHomography(pix, mm, 0)
assert H is not None
def project(matrix, p):
    return cv2.perspectiveTransform(np.float64(p).reshape(-1, 1, 2), matrix).reshape(-1, 2)
errors = np.linalg.norm(project(H, pix) - mm, axis=1)
loo = []
for k in range(len(pix)):
    keep = np.arange(len(pix)) != k
    hk, _ = cv2.findHomography(pix[keep], mm[keep], 0)
    assert hk is not None
    loo.append(float(np.linalg.norm(project(hk, pix[k:k+1])[0] - mm[k])))
# Affine robot mapping is a provisional interpolant of inconsistent A/B/C readings.
# It is not an independently validated rigid transform or TCP calibration.
grid_abc = np.float32([[0, 0], [100, -175], [100, 0]])
robot_abc = np.float32([[150.9, -71.8], [-21, -158.9], [148.8, -170.8]])
affine = cv2.getAffineTransform(grid_abc, robot_abc)
robot_h = np.vstack([affine, [0, 0, 1]]) @ H
annotated = im.copy()
for p, q in points:
    xy = tuple(np.rint(p).astype(int))
    cv2.circle(annotated, xy, 7, (0, 255, 0), 1)
    cv2.putText(annotated, '%d,%d' % tuple(q), (xy[0]+5, xy[1]-5), cv2.FONT_HERSHEY_SIMPLEX, .32, (0, 255, 0), 1)
for label, p in zip(['A?', 'C?', 'B?'], seed_pix):
    xy = tuple(p.astype(int))
    cv2.circle(annotated, xy, 12, (0, 255, 255), 2)
    cv2.putText(annotated, label, (xy[0]+12, xy[1]), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 255), 2)
cv2.imwrite(PREFIX + '_review.jpg', annotated)
result = {
    'status': 'provisional_requires_grid_identity_review_and_independent_robot_validation',
    'source': SOURCE, 'resolution': [640, 480], 'undistorted': True,
    'camera': 'wrist', 'device': '/dev/v4l/by-path/platform-70090000.xusb-usb-0:2.2:1.0-video-index0',
    'capture_robot_coords': [63.8, -113.2, 286.7, 174.86, -2.22, -98.11],
    'validity': 'Only this fixed camera pose, table plane, and hole convex hull; block height ignored at user request.',
    'grid_axes': 'origin A; right=A->C; inward=B->C; mm',
    'H_pixel_to_grid_mm': H.tolist(),
    'H_pixel_to_robot_xy_PROVISIONAL': robot_h.tolist(),
    'pixel': pix.tolist(), 'grid_mm': mm.tolist(),
    'hole_count': len(points), 'fit_mean_mm': float(errors.mean()),
    'leave_one_out_mean_mm': float(np.mean(loo)), 'leave_one_out_max_mm': max(loo),
    'caution': 'Grid errors only validate internal hole consistency, not robot accuracy, grid identities, screw alignment or robot kinematics. Robot transform uses an affine fit to three measured points; do not command motion from it.',
}
with open(PREFIX + '.json', 'x') as f:
    json.dump(result, f, indent=2)
print(json.dumps({k: result[k] for k in ['status', 'hole_count', 'fit_mean_mm', 'leave_one_out_mean_mm', 'leave_one_out_max_mm']}))
