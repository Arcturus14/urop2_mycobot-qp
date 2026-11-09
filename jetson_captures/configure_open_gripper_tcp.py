"""Apply USER REQUESTED nominal max-open fingertip-midpoint TCP. NO motion.

Model source: assets/official/gripper_contact_report.json controller_rad=0.15.
Physical maximum is user-reported; gripper API value is NOT assumed to be 100.
This is a model estimate, not measured TCP calibration.
"""
import json
import time
import math
import numpy as np
from pymycobot import MyCobot

tcp = [-0.6, 15.7, 104.3, 0, 0, 0]
prefix = 'tcp_max_open_' + time.strftime('%Y%m%d_%H%M%S')
mc = MyCobot('/dev/ttyTHS1', 1000000)
def snapshot():
    return {key: getattr(mc, key)() for key in (
        'get_end_type', 'get_reference_frame', 'get_tool_reference',
        'get_world_reference', 'get_gripper_value', 'get_angles', 'get_coords')}
def tip_from_flange(pose):
    rx, ry, rz = map(math.radians, pose[3:])
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    rotation = np.array([[cz*cy, cz*sy*sx-sz*cx, cz*sy*cx+sz*sx],
                         [sz*cy, sz*sy*sx+cz*cx, sz*sy*cx-cz*sx],
                         [-sy, cy*sx, cy*cx]])
    return (np.array(pose[:3]) + rotation @ np.array(tcp[:3])).tolist()
try:
    time.sleep(1)
    before = snapshot()
    assert before['get_end_type'] == 0, 'Unexpected endpoint; stop'
    assert before['get_reference_frame'] == 0, 'Unexpected frame; stop'
    assert before['get_world_reference'] == [0, 0, 0, 0, 0, 0], 'Unexpected world transform; stop'
    assert isinstance(before['get_coords'], list) and len(before['get_coords']) == 6
    with open(prefix + '_before.json', 'x') as f:
        json.dump(before, f, indent=2)
    applied = False
    try:
        mc.set_tool_reference(tcp)
        applied = True
        time.sleep(.4)
        actual = mc.get_tool_reference()
        assert isinstance(actual, list) and len(actual) == 6 and max(abs(a-b) for a,b in zip(actual,tcp)) < .2, str(actual)
        mc.set_end_type(1)
        time.sleep(.4)
        after = snapshot()
        assert after['get_end_type'] == 1, 'Tool endpoint not selected'
        assert after['get_reference_frame'] == 0, 'Base frame changed unexpectedly'
        assert max(abs(a-b) for a,b in zip(before['get_angles'], after['get_angles'])) < .5, 'Joints changed during check'
        predicted = tip_from_flange(before['get_coords'])
        difference = float(np.linalg.norm(np.array(after['get_coords'][:3]) - predicted))
        assert difference < 2, 'Firmware TCP result inconsistent with model transform: ' + str(difference)
    except Exception:
        if applied:
            mc.set_tool_reference(before['get_tool_reference'])
            time.sleep(.3)
            mc.set_end_type(before['get_end_type'])
        raise
    old_f = [157.6, 101.7, 89.2, 179.78, 1.83, -47.19]
    result = {'tcp_status': 'nominal_max_open_MODEL_ESTIMATE_NOT_MEASURED_CALIBRATION',
              'tcp': tcp, 'before': before, 'after': after,
              'firmware_vs_expected_xyz_mm': difference,
              'historical_F_flange': old_f,
              'historical_F_tip_xyz_computed_not_new_measurement': tip_from_flange(old_f),
              'robot_motion_commanded': False,
              'note': 'Coordinate command interpretation now uses tool endpoint. Old flange-coordinate commands must not be reused unchanged. Applies only to modeled maximum opening and standard mounting. Surface Z was not reset.'}
    with open(prefix + '_result.json', 'x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps({'files_prefix': prefix, **result}))
finally:
    mc.close()
