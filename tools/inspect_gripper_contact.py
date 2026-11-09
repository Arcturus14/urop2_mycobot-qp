"""Offline inspection of official fingertip meshes and a recorded floor contact."""
from pathlib import Path
import json
import sys
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mycobot_id.kinematics import _origin_matrix, _rotation_z, forward_kinematics


def mesh_points(path):
    root = ET.parse(path).getroot()
    ns = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
    ids = {n.get("id"): n for n in root.iter() if n.get("id")}
    unit = float(root.find("c:asset/c:unit", ns).get("meter"))
    parts = []
    def walk(node, matrix):
        for child in node:
            tag = child.tag.split("}")[-1]
            if tag == "matrix":
                matrix = matrix @ np.fromstring(child.text, sep=" ").reshape(4,4)
            elif tag in ("rotate", "translate", "scale"):
                raise ValueError("Unsupported COLLADA transform")
            elif tag == "node":
                walk(child, matrix.copy())
            elif tag == "instance_node":
                walk(ids[child.get("url")[1:]], matrix.copy())
            elif tag == "instance_geometry":
                g = ids[child.get("url")[1:]]
                v = g.find("c:mesh/c:vertices/c:input[@semantic='POSITION']", ns)
                s = ids[v.get("source")[1:]]
                accessor = s.find("c:technique_common/c:accessor", ns)
                data = np.fromstring(ids[accessor.get("source")[1:]].text, sep=" ")
                p = data.reshape(-1, int(accessor.get("stride")))[:,:3]
                parts.append((p @ matrix[:3,:3].T + matrix[:3,3]) * unit)
    ref = root.find("c:scene/c:instance_visual_scene", ns).get("url")[1:]
    walk(ids[ref], np.eye(4))
    return np.vstack(parts)


def inspect():
    # Source URDF contains a malformed limit attribute; repair ONLY that XML typo
    # in memory so transforms and geometry remain exactly as published.
    text = (ROOT / "assets/official/gripper_mount.urdf").read_text()
    urdf = ET.fromstring(text.replace('lower = "-2.932"1', 'lower = "-2.9321"'))
    def origin(e):
        return _origin_matrix(np.fromstring(e.get("xyz", "0 0 0"), sep=" "),
                              np.fromstring(e.get("rpy", "0 0 0"), sep=" "))
    mount = origin(urdf.find("joint[@name='joint6output_to_gripper_base']/origin"))
    contact = json.loads((ROOT / "data/calibration/floor_contact_20261104_1906.json").read_text())
    flange = forward_kinematics(contact["joints_deg"]).transforms[-1]
    report = {"samples": [], "note": "Official JN slider maps API 0 to controller -0.74 rad and API 100 to 0.15 rad."}
    closed_points = []
    for angle in (-0.74, 0.0, 0.15):
        item = {"controller_rad": angle, "fingers": {}}
        for side, sign in (("left", 1), ("right", -1)):
            first = urdf.find("joint[@name='gripper_controller']" if side == "left" else "joint[@name='gripper_base_to_gripper_right3']")
            second = urdf.find(f"joint[@name='gripper_{side}3_to_gripper_{side}1']")
            r1, r2 = np.eye(4), np.eye(4)
            r1[:3,:3] = _rotation_z(sign * angle)
            r2[:3,:3] = _rotation_z(-sign * angle)
            visual = origin(urdf.find(f"link[@name='gripper_{side}1']/visual/origin"))
            transform = mount @ origin(first.find("origin")) @ r1 @ origin(second.find("origin")) @ r2 @ visual
            p = mesh_points(ROOT / f"assets/official/gripper_{side}1.dae")
            p = p @ transform[:3,:3].T + transform[:3,3]
            world = p @ flange[:3,:3].T + flange[:3,3]
            # Average the outermost axial tip surface, not the collision box.
            distal = p[p[:,2] >= p[:,2].max() - .00005]
            if angle == -0.74:
                closed_points.append(distal.mean(axis=0))
            item["fingers"][side] = {"min_flange_mm": (p.min(axis=0)*1000).tolist(),
                                     "max_flange_mm": (p.max(axis=0)*1000).tolist(),
                                     "distal_center_flange_mm": (distal.mean(axis=0)*1000).tolist(),
                                     "lowest_world_z_mm": float(world[:,2].min()*1000)}
        report["samples"].append(item)
    from OCP.STEPControl import STEPControl_Reader
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    r = STEPControl_Reader()
    r.ReadFile(str(ROOT / "assets/official/adaptive_gripper_230708.step"))
    r.TransferRoots()
    box = Bnd_Box()
    BRepBndLib.AddOptimal_s(r.OneShape(), box, False, False)
    report["step_bounds_mm"] = list(box.Get())
    tcp = np.mean(closed_points, axis=0)
    tcp_world = flange[:3,:3] @ tcp + flange[:3,3]
    report["closed_tip_midpoint_flange_m"] = tcp.tolist()
    report["contact_tcp_raw_xyz_mm"] = (tcp_world*1000).tolist()
    report["model_to_table_z_offset_m"] = -float(tcp_world[2])
    report["closed_tip_points_flange_m"] = [p.tolist() for p in closed_points]
    report["calibration_status"] = "single-contact provisional Z translation; not full FK calibration"
    output = json.dumps(report, indent=2)
    (ROOT / "assets/official/gripper_contact_report.json").write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    inspect()
