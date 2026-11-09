"""Correct D/F identities and recompute XY from the actual F anchor.

Uses the saved image plane fit only; no camera/robot access.
User correction: F=upper left, E=upper right, D=lower right.
Consequently F->E is inward 150 mm; E->D is right 175 mm.
"""
import json
import cv2
import numpy as np

OLD = 'DEF_red_table_solution_v1.json'
PREFIX = 'DEF_red_table_solution_v2_corrected_labels'
with open(OLD) as f:
    old = json.load(f)
im = cv2.imread(old['source'])
assert im is not None
Q = np.float64(old['grid_right_inward_to_base_xy_rows'])
anchor = np.float64(old['xyz_mm']['F'][:2])
# Old local coordinates were relative to the incorrectly labeled lower-right F.
# New right = 175 - old inward; new inward = 150 - old right.
T = np.array([[0.,-1.,175.],[-1.,0.,150.],[0.,0.,1.]])
H = T @ np.float64(old['H_pixel_to_F_grid_mm'])
def project(matrix, points):
    return cv2.perspectiveTransform(np.float64(points).reshape(-1,1,2),matrix).reshape(-1,2)
def xyz(g):
    return [*map(float, np.float64(g) @ Q + anchor), 0.]
red_grid = project(T, [old['red_from_F_right_inward_mm']])[0]
locations = {'F':[0,0], 'E':[0,150], 'D':[175,150]}
positions = {label:xyz(g) for label,g in locations.items()}
positions['red'] = xyz(red_grid)
result = dict(old)
result.update({
    'supersedes': OLD,
    'correction': 'User corrected D/F labels. F is upper-left, E upper-right, D lower-right. Anchor and image-grid axes were corrected, not just names.',
    'confirmed_gaps':{'F_to_E_inward':6,'E_to_D_right':7},
    'xyz_mm':positions,
    'red_from_F_right_inward_mm':red_grid.tolist(),
    'H_pixel_to_F_grid_mm':H.tolist(),
    'F_anchor_source':'User F contact measurement belongs to upper-left bolt; historical flange F transformed using nominal max-open TCP, not remeasured.',
    'note':'Replaces v1 including its incorrect bolt identities and red XY. XY remains an unvalidated model/TCP/ABC-axis estimate. Z=0 is user-defined table height.'
})
for label,g in locations.items():
    p=project(np.linalg.inv(H),[g])[0]
    if 0<=p[0]<640 and 0<=p[1]<480:
        xy=tuple(np.rint(p).astype(int))
        cv2.circle(im,xy,10,(0,255,255),2)
        cv2.putText(im,label,(xy[0]+10,xy[1]-8),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,255,255),2)
p=old['red_center_pixel']; xy=tuple(np.rint(p).astype(int))
cv2.drawMarker(im,xy,(0,255,255),cv2.MARKER_CROSS,18,2)
cv2.putText(im,'XY~(%.1f, %.1f), table Z=0'%tuple(positions['red'][:2]),(195,xy[1]-45),cv2.FONT_HERSHEY_SIMPLEX,.43,(0,255,0),1)
assert abs(np.linalg.norm(np.array(positions['E'])-positions['F'])-150)<1e-6
assert abs(np.linalg.norm(np.array(positions['D'])-positions['E'])-175)<1e-6
with open(PREFIX+'.json','x') as f:
    json.dump(result,f,indent=2)
assert cv2.imwrite(PREFIX+'.jpg',im)
print(json.dumps({'xyz_mm':positions,'red_from_F_right_inward_mm':red_grid.tolist(),'files_prefix':PREFIX}))
