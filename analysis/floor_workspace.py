"""Numerical floor reach estimates, not a certified collision-free workspace."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

from mycobot_id.kinematics import JOINT_LIMITS_DEG, ToolGeometry, forward_kinematics, self_collision

ROOT = Path(__file__).resolve().parents[1]


def main():
    config = json.loads((ROOT / "configs/robot_pose_gui.json").read_text())
    tool = ToolGeometry(**config["tool"])
    cal = config["floor_calibration"]
    tcp = np.r_[cal["closed_tip_midpoint_flange_m"], 1.]
    offset = cal["model_to_table_z_offset_m"]
    limits = np.radians(JOINT_LIMITS_DEG)
    def pose(q):
        t = forward_kinematics(np.degrees(q), tool).transforms[-1]
        p = (t @ tcp)[:3]
        p[2] += offset
        return t, p
    def equality(q):
        t, p = pose(q)
        return np.array([p[2], t[0,2], t[1,2]])
    def direction(q):
        return -pose(q)[0][2,2] - .99
    rng = np.random.default_rng(20261104)
    seeds = [np.radians([-55.01,-33.48,-109.86,53.87,1.31,-33.48]),
             np.radians([36.82,-54.84,-66.53,33.48,-1.31,-58.35])]
    seeds += [rng.uniform(limits[:,0], limits[:,1]) for _ in range(18)]
    results = {}
    for name in ("radius_max", "radius_min", "x_min", "x_max", "y_min", "y_max"):
        accepted = []
        def objective(q):
            p = pose(q)[1]
            if name.startswith("radius"):
                value = np.linalg.norm(p[:2])
            else:
                value = p[0 if name.startswith("x") else 1]
            return -value if name.endswith("max") else value
        for seed in seeds:
            fit = minimize(objective, seed, method="SLSQP", bounds=limits,
                           constraints=[{"type":"eq", "fun":equality},
                                        {"type":"ineq", "fun":direction}],
                           options={"maxiter":180,"ftol":1e-10})
            q = fit.x
            if not fit.success or np.max(np.abs(equality(q))) > 1e-5 or direction(q) < -1e-5:
                continue
            if self_collision(np.degrees(q), tool) is not None:
                continue
            t, p = pose(q)
            accepted.append({"value":float(objective(q)), "xyz_mm":(p*1000).tolist(),
                             "radius_mm":float(np.linalg.norm(p[:2])*1000),
                             "joints_deg":np.degrees(q).tolist()})
        results[name] = min(accepted, key=lambda x:x["value"]) if accepted else None
        print(name, results[name], flush=True)
    report = {"method":"20-seed SLSQP extrema; final poses filtered by configured self-collision model",
              "reference":"URDF XY; table Z=0 after provisional contact correction",
              "tool":"closed fingertip midpoint, flange +Z down; yaw free",
              "limitations":["Not exhaustive or globally certified", "No table/case/environment collision model",
                             "No path accessibility check", "Bounds do not define a fully reachable rectangle"],
              "results":results}
    target = ROOT / "reports/floor_workspace.json"
    target.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
