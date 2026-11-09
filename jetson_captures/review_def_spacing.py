"""Read-only photo analysis: count holes between labeled bolts. No robot access."""
import json
import cv2
import numpy as np

source = 'wrist_new_bolts_20261104_125432.jpg'
im = cv2.imread(source)
assert im is not None
_, mask = cv2.threshold(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), 90, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
centers = []
for c in contours:
    a = cv2.contourArea(c)
    x, y, w, h = cv2.boundingRect(c)
    if 8 < a < 180 and .5 < w/h < 1.8:
        m = cv2.moments(c)
        centers.append(np.array([m['m10']/m['m00'], m['m01']/m['m00']]))
bolts = {'A': [311, 354], 'D': [283, 100], 'E': [490, 85], 'F': [528, 326]}
results = {}
for start, end in [('F','E'), ('E','D')]:
    p, q = np.array(bolts[start]), np.array(bolts[end])
    v = q-p
    selected = []
    for center in centers:
        t = np.dot(center-p, v)/np.dot(v, v)
        distance = np.linalg.norm(center-p-t*v)
        if .045 < t < .955 and distance < 10:
            selected.append((float(t), center))
    selected.sort(key=lambda item: item[0])
    results[start+'_'+end] = {'intermediate_holes': len(selected),
        'implied_gaps': len(selected)+1,
        'centers': [c.tolist() for _,c in selected]}
    cv2.line(im, tuple(p), tuple(q), (0,255,255), 1)
    for i, (_, center) in enumerate(selected, 1):
        xy = tuple(np.rint(center).astype(int))
        cv2.circle(im, xy, 6, (0,255,0), 1)
        cv2.putText(im, str(i), (xy[0]+7, xy[1]), cv2.FONT_HERSHEY_SIMPLEX, .4, (0,255,0), 1)
for label, pixel in bolts.items():
    cv2.circle(im, tuple(pixel), 10, (0,255,255), 2)
    cv2.putText(im,label,(pixel[0]-15,pixel[1]-15),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,255,255),2)
assert cv2.imwrite('DEF_spacing_review.jpg', im)
with open('DEF_spacing_review.json','w') as f:
    json.dump({'source':source,'observations':results,'user_requested_gaps':{'F_E':6,'E_D':7},
               'note':'Counts assume currently labeled bolts are the intended targets; review required.'},f,indent=2)
print(json.dumps(results))
