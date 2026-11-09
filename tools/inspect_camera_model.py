"""Read the official COLLADA scene, including instances, units and URDF origins."""
from pathlib import Path
import json
import hashlib
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mycobot_id.kinematics import _origin_matrix


def inspect():
    path = ROOT / "assets/official/camera_flange.dae"
    root = ET.parse(path).getroot()
    ns = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
    ids = {node.get("id"): node for node in root.iter() if node.get("id")}
    unit = float(root.find("c:asset/c:unit", ns).get("meter"))
    if root.findtext("c:asset/c:up_axis", namespaces=ns) != "Z_UP":
        raise ValueError("unexpected scene up axis")
    parts = []

    def walk(node, transform, stack=()):
        if node.get("id") in stack:
            raise ValueError("cyclic scene instance")
        stack = (*stack, node.get("id"))
        for child in node:
            tag = child.tag.split("}")[-1]
            if tag == "matrix":
                # Rhino writes these matrices row-major (translation in 4,8,12).
                transform = transform @ np.fromstring(child.text, sep=" ").reshape(4, 4)
            elif tag in ("translate", "rotate", "scale"):
                raise ValueError("unsupported scene transform: " + tag)
            elif tag == "node":
                walk(child, transform.copy(), stack)
            elif tag == "instance_node":
                walk(ids[child.get("url")[1:]], transform.copy(), stack)
            elif tag == "instance_geometry":
                geometry = ids[child.get("url")[1:]]
                vertices = geometry.find("c:mesh/c:vertices/c:input[@semantic='POSITION']", ns)
                source = ids[vertices.get("source")[1:]]
                accessor = source.find("c:technique_common/c:accessor", ns)
                stride = int(accessor.get("stride", "1"))
                array = ids[accessor.get("source")[1:]]
                values = np.fromstring(array.text, sep=" ").reshape(-1, stride)[:, :3]
                points = (values @ transform[:3, :3].T + transform[:3, 3]) * unit
                parts.append(points)

    scene_ref = root.find("c:scene/c:instance_visual_scene", ns).get("url")[1:]
    walk(ids[scene_ref], np.eye(4))
    points = np.vstack(parts)
    low, high = points.min(axis=0), points.max(axis=0)
    urdf = ET.parse(ROOT / "assets/official/camera_mount.urdf").getroot()
    joint = urdf.find("joint[@name='joint6output_to_camera_flange']/origin")
    mesh = urdf.find("link[@name='camera_flange']/collision/origin")
    def origin(element):
        return _origin_matrix(np.fromstring(element.get("xyz"), sep=" "), np.fromstring(element.get("rpy"), sep=" "))
    combined = origin(joint) @ origin(mesh)
    center = combined @ np.r_[(low + high) / 2, 1.0]
    rotation = combined[:3, :3]
    rpy = np.degrees([
        np.arctan2(rotation[2, 1], rotation[2, 2]),
        np.arctan2(-rotation[2, 0], np.hypot(rotation[0, 0], rotation[1, 0])),
        np.arctan2(rotation[1, 0], rotation[0, 0]),
    ])
    report = {
        "unit_m": unit, "parts": len(parts), "vertices": len(points),
        "mesh_min_m": low.tolist(), "mesh_max_m": high.tolist(),
        "box_size_m": (high-low).tolist(), "box_center_flange_m": center[:3].tolist(),
        "box_rotation_flange": combined[:3, :3].tolist(),
        "part_bounds_m": [[p.min(axis=0).tolist(), p.max(axis=0).tolist()] for p in parts],
        "joint_xyz": joint.get("xyz"), "joint_rpy": joint.get("rpy"),
        "mesh_xyz": mesh.get("xyz"), "mesh_rpy": mesh.get("rpy"),
        "box_rpy_deg": rpy.tolist(),
        "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            path, ROOT / "assets/official/camera_mount.urdf",
            ROOT / "assets/official/camera_flange_230708.step",
        )},
    }
    # OCP is an optional development-only dependency, not needed by the GUI.
    try:
        from OCP.STEPControl import STEPControl_Reader
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.Bnd import Bnd_Box
        from OCP.BRepBndLib import BRepBndLib
        reader = STEPControl_Reader()
        status = reader.ReadFile(str(ROOT / "assets/official/camera_flange_230708.step"))
        if status != IFSelect_RetDone:
            raise ValueError("STEP read failed")
        reader.TransferRoots()
        bounds = Bnd_Box()
        BRepBndLib.AddOptimal_s(reader.OneShape(), bounds, False, False)
        report["step_bounds_mm"] = list(bounds.Get())
        np.testing.assert_allclose(np.array(bounds.Get()) / 1000, np.r_[low, high], atol=2e-8)
        report["step_matches_dae_bounds"] = True
    except ImportError:
        report["step_matches_dae_bounds"] = None
    output = json.dumps(report, indent=2)
    if "--report" in sys.argv:
        (ROOT / "assets/official/camera_geometry_report.json").write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    inspect()
