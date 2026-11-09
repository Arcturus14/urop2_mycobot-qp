"""Offline table geometry, anchored at F. No camera/serial access or motion.

XY base axes; Z=0 is the USER DEFINED table height, not base-Z measurement.
All robot XY estimates inherit model TCP and earlier A/B/C errors.
"""
import json
import argparse
from pathlib import Path
import math
import cv2
import numpy as np

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source',default='red_table_check_20261104_131901.jpg')
parser.add_argument('--prefix',default='DEF_red_table_solution_v1')
parser.add_argument('--seed-pixels',type=float,nargs=6,default=[525,374,423,70,253,453],help='Lower-right, upper-right, lower-left bolt image centers')
parser.add_argument('--corrected-def',action='store_true',help='F upper-left, E upper-right, D lower-right')
args=parser.parse_args()
SOURCE=args.source
PREFIX=args.prefix
if Path(PREFIX+'.json').exists() or Path(PREFIX+'.jpg').exists():
    raise SystemExit('Choose a new output prefix; existing outputs are preserved')
TCP = np.array([-.6, 15.7, 104.3])
def tip(pose):
    rx, ry, rz = np.radians(pose[3:])
    cx,sx,cy,sy,cz,sz = math.cos(rx),math.sin(rx),math.cos(ry),math.sin(ry),math.cos(rz),math.sin(rz)
    rot = np.array([[cz*cy,cz*sy*sx-sz*cx,cz*sy*cx+sz*sx],
                    [sz*cy,sz*sy*sx+cz*cx,sz*sy*cx-cz*sx],[-sy,cy*sx,cy*cx]])
    return np.array(pose[:3]) + rot @ TCP
abc_flange = [[150.9,-71.8,89.7,177.95,2.59,-81.65],
              [-21,-158.9,91.6,177.09,-.91,-134.84],
              [148.8,-170.8,90.3,178.27,2.56,-122.59]]
abc_tcp = np.array([tip(p)[:2] for p in abc_flange])
# Right and inward coordinates from the original S-based definitions, mm.
abc_grid = np.array([[0,200],[100,25],[100,200]],dtype=float)
g0,b0 = abc_grid.mean(axis=0),abc_tcp.mean(axis=0)
u,_,vt = np.linalg.svd((abc_grid-g0).T @ (abc_tcp-b0))
Q = u @ vt
assert np.linalg.det(Q)>0, 'Unexpected axis handedness'
abc_residual = np.linalg.norm((abc_grid-g0) @ Q+b0-abc_tcp,axis=1)
f_flange = [157.6,101.7,89.2,179.78,1.83,-47.19]
f_xy = tip(f_flange)[:2]
def base(grid):
    return np.array(grid) @ Q + f_xy
im = cv2.imread(SOURCE)
assert im is not None
# Screw heads seed integer identities only; plane fitting uses hole centers.
seed_pixel=np.float32(args.seed_pixels).reshape(3,2)
seed_grid=np.float32([[7,6],[0,6],[7,0]] if args.corrected_def else [[0,0],[0,7],[6,0]])
initial=cv2.getAffineTransform(seed_pixel,seed_grid)
_,mask=cv2.threshold(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),90,255,cv2.THRESH_BINARY_INV)
contours,_=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
points=[]
all_centers=[]
for c in contours:
    area=cv2.contourArea(c)
    x,y,w,h=cv2.boundingRect(c)
    if not (12<area<220 and .55<w/h<1.7 and x>220 and 15<y<463): continue
    m=cv2.moments(c)
    p=np.array([m['m10']/m['m00'],m['m01']/m['m00']])
    if np.min(np.linalg.norm(seed_pixel-p,axis=1))<18: continue
    all_centers.append(p)
    g=initial @ np.r_[p,1]
    index=np.rint(g)
    if np.linalg.norm(g-index)<.25 and -2<=index[0]<=9 and -2<=index[1]<=9:
        points.append((p,index*25))
assert len(points)>=20, 'Too few grid correspondences'
unique={}
for p,g in points:
    key=tuple(g)
    error=np.linalg.norm(initial @ np.r_[p,1]-g/25)
    if key not in unique or error<unique[key][0]:
        unique[key]=(error,p,g)
points=[(p,g) for _,p,g in unique.values()]
assert len(points)>=20, 'Too few unique grid correspondences'
pix=np.float64([p for p,_ in points]); grid=np.float64([g for _,g in points])
candidate_count=len(points)
_,inliers=cv2.findHomography(pix,grid,cv2.RANSAC,1.0)
assert inliers is not None
points=[p for p,keep in zip(points,inliers.ravel()) if keep]
assert len(points)>=20, 'Too few consistent hole centers'
pix=np.float64([p for p,_ in points]); grid=np.float64([g for _,g in points])
H,_=cv2.findHomography(pix,grid,0)
def project(matrix,p):
    return cv2.perspectiveTransform(np.float64(p).reshape(-1,1,2),matrix).reshape(-1,2)
# The seed affine is only approximate. Expand using the validated perspective fit.
expanded={}
for p,g in zip(all_centers,project(H,all_centers)):
    index=np.rint(g/25)
    error=np.linalg.norm(g-index*25)
    if error<2.5 and np.all(index>=-2) and np.all(index<=9):
        key=tuple(index)
        if key not in expanded or error<expanded[key][0]:
            expanded[key]=(error,p,index*25)
points=[(p,g) for _,p,g in expanded.values()]
pix=np.float64([p for p,_ in points]); grid=np.float64([g for _,g in points])
_,inliers=cv2.findHomography(pix,grid,cv2.RANSAC,1.0)
assert inliers is not None
points=[p for p,keep in zip(points,inliers.ravel()) if keep]
assert len(points)>=20
pix=np.float64([p for p,_ in points]); grid=np.float64([g for _,g in points])
H,_=cv2.findHomography(pix,grid,0)
errors=np.linalg.norm(project(H,pix)-grid,axis=1)
loo=[]
for k in range(len(points)):
    keep=np.arange(len(points))!=k
    hk,_=cv2.findHomography(pix[keep],grid[keep],0)
    loo.append(float(np.linalg.norm(project(hk,pix[k:k+1])[0]-grid[k])))
assert max(loo)<2, 'Grid fit failed validation'
hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
red=cv2.inRange(hsv,(0,90,45),(12,255,255))|cv2.inRange(hsv,(168,90,45),(179,255,255))
roi=np.zeros(im.shape[:2],np.uint8)
cv2.fillConvexPoly(roi,np.rint(cv2.convexHull(pix.astype(np.float32))).astype(np.int32),255)
red &= roi
cs,_=cv2.findContours(red,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cs=[c for c in cs if cv2.contourArea(c)>100]
assert len(cs)==1, 'Expected exactly one red patch in calibrated hull'
c=cs[0]
distance=cv2.distanceTransform(roi,cv2.DIST_L2,3)
xy=c.reshape(-1,2)
assert np.min(distance[xy[:,1],xy[:,0]])>2, 'Red patch clipped by calibrated area'
transformed=project(H,c).reshape(-1,1,2).astype(np.float32)
m=cv2.moments(transformed)
red_grid=np.array([m['m10']/m['m00'],m['m01']/m['m00']])
center=project(np.linalg.inv(H),[red_grid])[0]
red_base=base(red_grid)
locations={'F':[0,0],'E':[0,150],'D':[175,150]} if args.corrected_def else {'F':[0,0],'E':[0,175],'D':[150,175]}
positions={name:[*map(float,base(g)),0.] for name,g in locations.items()}
positions['red']=[*map(float,red_base),0.]
for p,g in points:
    cv2.circle(im,tuple(np.rint(p).astype(int)),3,(0,180,0),1)
cv2.drawContours(im,[c],-1,(0,255,0),2)
ctr=tuple(np.rint(center).astype(int))
cv2.drawMarker(im,ctr,(0,255,255),cv2.MARKER_CROSS,15,2)
cv2.putText(im,'XY~(%.1f, %.1f), table Z=0'%tuple(red_base),(200,ctr[1]-45),cv2.FONT_HERSHEY_SIMPLEX,.43,(0,255,255),1)
for name,g in locations.items():
    p=project(np.linalg.inv(H),[g])[0]
    if 0<=p[0]<640 and 0<=p[1]<480:
        xy=tuple(np.rint(p).astype(int)); cv2.circle(im,xy,9,(0,255,255),1)
        cv2.putText(im,name,(xy[0]+10,xy[1]),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,255),2)
result={
 'status':'PROVISIONAL_MODEL_TCP_AND_ABC_AXIS_ESTIMATE_NO_INDEPENDENT_ROBOT_VALIDATION',
 'source':SOURCE,'xyz_convention':'XY in robot base axes; Z is table-relative defined zero, NOT base Cartesian Z',
 'confirmed_gaps':{'F_to_E_inward':6,'E_to_D_right':7} if args.corrected_def else {'F_to_E_inward':7,'E_to_D_right':6},'pitch_mm':25,
 'tcp_model_offset_mm':TCP.tolist(),'abc_flange':abc_flange,'abc_tcp_xy':abc_tcp.tolist(),
 'grid_right_inward_to_base_xy_rows':Q.tolist(),'abc_rigid_fit_residual_mm':abc_residual.tolist(),
 'F_anchor_source':'historical flange F transformed using nominal max-open TCP, not remeasured',
 'xyz_mm':positions,'red_from_F_right_inward_mm':red_grid.tolist(),'red_center_pixel':center.tolist(),
 'H_pixel_to_F_grid_mm':H.tolist(),'hole_count':len(points),'hole_candidates':candidate_count,'grid_fit_mean_mm':float(errors.mean()),
 'grid_loo_mean_mm':float(np.mean(loo)),'grid_loo_max_mm':max(loo),
 'camera_snapshot_pose_tcp':json.loads(Path(SOURCE).with_suffix('.json').read_text())['coords'],
 'validity':'Saved snapshot only; camera must not move without refitting. Object height ignored.',
 'robot_motion_commanded':False,
 'note':'Internal grid residuals do not establish absolute robot accuracy. Replaces old affine XY estimate for this snapshot only.'}
with open(PREFIX+'.json','x') as f: json.dump(result,f,indent=2)
assert cv2.imwrite(PREFIX+'.jpg',im)
print(json.dumps(result))
