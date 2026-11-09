"""Laptop SSH deployment and data retrieval; standard library only.

Acquisition runs on the Jetson. The identify action requires explicit motion confirmation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import tarfile
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / ".jetson" / "state.json"


def ssh(host, command, **kwargs):
    """Run SSH and ensure Ctrl+C closes the session, including on Windows.

    ``subprocess.run`` can leave the OpenSSH child alive while Python unwinds a
    KeyboardInterrupt.  The server then keeps the foreground command (and the
    robot lock) alive.  Owning the Popen object lets us close that connection
    deterministically; the Jetson-side controller receives SIGHUP and stops.
    """
    command_line = ["ssh", "-T", "-o", "BatchMode=yes", "-o",
                    "ConnectTimeout=10", "-o", "StrictHostKeyChecking=yes",
                    host, command]
    check = kwargs.pop("check", True)
    timeout = kwargs.pop("timeout", None)
    input_data = kwargs.pop("input", None)
    if kwargs.pop("capture_output", False):
        if "stdout" in kwargs or "stderr" in kwargs:
            raise ValueError("stdout/stderr may not be used with capture_output")
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    process = subprocess.Popen(command_line, **kwargs)
    try:
        stdout, stderr = process.communicate(input=input_data, timeout=timeout)
    except KeyboardInterrupt:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        raise subprocess.TimeoutExpired(command_line, timeout, output=stdout, stderr=stderr)
    result = subprocess.CompletedProcess(command_line, process.returncode, stdout, stderr)
    if check:
        result.check_returncode()
    return result


def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]


def load_state(config, state_file=STATE):
    if not state_file.is_file():
        raise SystemExit("Deploy to this target first")
    state = json.loads(state_file.read_text(encoding="utf-8"))
    if state["config"] != config:
        raise SystemExit("Connection config changed; deploy to this target first")
    release = PurePosixPath(state["release"])
    if release.parent != PurePosixPath(config["remote_root"]) / "releases":
        raise SystemExit("Invalid deployment state")
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["check", "deploy", "collect", "identify", "fetch",
                                           "qp-deploy", "qp-run", "qp-fetch"])
    parser.add_argument("--config", default="configs/jetson.json")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--poll-interval", type=float, default=0.05)
    parser.add_argument("--confirm-motion", default="")
    parser.add_argument("--joints", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--reference-angles", type=float, nargs=6)
    parser.add_argument("--goal-mm", type=float, nargs=3, metavar=("X", "Y", "Z"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--b-approach", action="store_true")
    mode.add_argument("--b-hover", action="store_true")
    mode.add_argument("--hover-over-mm", type=float, nargs=3, metavar=("X", "Y", "Z"))
    mode.add_argument("--bs-midpoint-hover", action="store_true")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--profile", action="store_true", help="Record QP process CPU and peak RSS")
    parser.add_argument("--obstacle-center-mm", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--via-mm", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--reference-posture-deg", type=float, nargs=6)
    parser.add_argument("--detour-j1", action="store_true")
    parser.add_argument("--grasp-direction", choices=("vertical", "horizontal"), default="vertical")
    parser.add_argument("--grasp-angle-deg", type=float, default=0.0,
                        help="For horizontal grasping, angle from workspace +X toward +Y")
    parser.add_argument("--period", type=float, help="QP period in seconds")
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    host = config["ssh_host"]
    base = config["remote_root"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@-]*", host):
        parser.error("Invalid SSH alias")
    if not re.fullmatch(r"/home/[A-Za-z0-9_.-]+/(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+", base):
        parser.error("remote_root must be a project directory beneath /home/<user>/")
    if ".." in PurePosixPath(base).parts:
        parser.error("Invalid remote_root")
    qp = args.action.startswith("qp-")
    state_file = STATE
    if qp:
        base = str(PurePosixPath(base).with_name("urop2-osqp"))
        config = {**config, "remote_root": base}
        state_file = STATE.with_name("qp-state.json")
    if args.action == "check":
        ssh(host, "hostname && uname -m && python3 --version && "
            "python3 -m pip show pymycobot pyserial && ls -l " + shlex.quote(config["port"]))
        return
    if args.action in {"deploy", "qp-deploy"}:
        release = base + "/releases/" + stamp()
        # A new release directory means repeated deployments never overwrite code
        # used by an earlier experiment or touch its data.
        ssh(host, "mkdir -p " + shlex.quote(base + "/releases") + " && mkdir " + shlex.quote(release))
        with tempfile.TemporaryFile() as archive:
            with tarfile.open(fileobj=archive, mode="w:gz") as bundle:
                entries = (["mycobot_id", "qp_control", "configs/robot_pose_gui.json"] if qp else
                           ["mycobot_id", "experiments", "configs", "scripts",
                            "requirements-jetson.txt", "pyproject.toml", "README.md"])
                for entry in entries:
                    source = ROOT / entry
                    files = sorted(source.rglob("*")) if source.is_dir() else [source]
                    for path in files:
                        if path.is_file() and "__pycache__" not in path.parts:
                            bundle.add(path, arcname=path.relative_to(ROOT).as_posix(), recursive=False)
            archive.seek(0)
            ssh(host, "tar -xzf - -C " + shlex.quote(release), stdin=archive)
        env_dir = shlex.quote(base + "/.venv")
        python = shlex.quote(base + "/.venv/bin/python")
        if qp:
            setup = (
                f"test -x {python} || python3 -m venv --system-site-packages {env_dir}"
            )
            command = (f"cd {shlex.quote(release)} && ({setup}) && "
                       f"{python} -m pip install --only-binary=:all: -r qp_control/requirements.txt && "
                       f"{python} -m qp_control.run --self-test")
        else:
            command = "cd " + shlex.quote(release) + " && python3 scripts/remote_setup.py " + env_dir
        ssh(host, command)
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text(json.dumps({"config": config, "release": release}, indent=2), encoding="utf-8")
        print("Deployed:", release)
        return
    state = load_state(config, state_file)
    release = state["release"]
    if args.action == "qp-run":
        hover = args.b_hover or args.bs_midpoint_hover or args.hover_over_mm is not None
        b_mode = args.b_approach or hover
        if b_mode == (args.goal_mm is not None):
            parser.error("qp-run requires an approach, hover, or --goal-mm X Y Z")
        if args.preview and not b_mode:
            parser.error("--preview requires an approach or hover mode")
        if args.preview and args.profile:
            parser.error("--profile measures motion and cannot be used with --preview")
        if args.obstacle_center_mm is not None and not hover:
            parser.error("--obstacle-center-mm requires a hover mode")
        if args.bs_midpoint_hover and args.obstacle_center_mm is not None:
            parser.error("--bs-midpoint-hover is tuned for an unobstructed workspace")
        if args.detour_j1 and (not args.b_hover or args.obstacle_center_mm is None):
            parser.error("--detour-j1 requires --b-hover and --obstacle-center-mm")
        if args.via_mm is not None and args.hover_over_mm is None:
            parser.error("--via-mm requires --hover-over-mm")
        if args.grasp_direction == "horizontal" and not (
            args.hover_over_mm is not None or args.bs_midpoint_hover
        ):
            parser.error("--grasp-direction horizontal requires --hover-over-mm or --bs-midpoint-hover")
        if not args.preview and args.confirm_motion != "I_UNDERSTAND":
            parser.error("qp-run motion requires --confirm-motion I_UNDERSTAND")
        command = ["flock", "-n", base + "/robot.lock",
                   base + "/.venv/bin/python", "-m", "qp_control.run",
                   "--port", config["port"], "--baud", str(config["baud"]),
                   "--duration", str(args.duration)]
        if b_mode:
            if args.bs_midpoint_hover:
                command += ["--bs-midpoint-hover"]
            elif args.hover_over_mm is not None:
                command += ["--hover-over-mm", *map(str, args.hover_over_mm)]
            else:
                command += ["--b-hover" if args.b_hover else "--b-approach"]
            if args.preview:
                command += ["--preview"]
        else:
            command += ["--goal-mm", *map(str, args.goal_mm)]
        if args.period is not None:
            command += ["--period", str(args.period)]
        if args.profile:
            command += ["--profile"]
        if args.obstacle_center_mm is not None:
            command += ["--obstacle-center-mm", *map(str, args.obstacle_center_mm)]
        if args.via_mm is not None:
            command += ["--via-mm", *map(str, args.via_mm)]
        if args.reference_posture_deg is not None:
            command += ["--reference-posture-deg", *map(str, args.reference_posture_deg)]
        if args.detour_j1:
            command += ["--detour-j1"]
        command += ["--grasp-direction", args.grasp_direction,
                    "--grasp-angle-deg", str(args.grasp_angle_deg)]
        try:
            ssh(host, "cd " + shlex.quote(release) + " && exec " + shlex.join(command))
        except subprocess.CalledProcessError as exc:
            raise SystemExit(exc.returncode) from None
        return
    if args.action == "identify":
        if args.confirm_motion != "I_UNDERSTAND":
            parser.error("identify requires --confirm-motion I_UNDERSTAND")
        command = [base + "/.venv/bin/python", "-m", "experiments.run_identification",
                   "--backend", "real", "--port", config["port"], "--baud", str(config["baud"]),
                   "--confirm-motion", args.confirm_motion, "--joints", *map(str, args.joints)]
        if args.reference_angles:
            command += ["--reference-angles", *map(str, args.reference_angles)]
        ssh(host, "cd " + shlex.quote(release) + " && " + shlex.join(command))
        return
    if args.action == "collect":
        command = [base + "/.venv/bin/python", "scripts/remote_collect.py",
                   "--port", config["port"], "--baud", str(config["baud"]),
                   "--samples", str(args.samples), "--duration", str(args.duration),
                   "--poll-interval", str(args.poll_interval)]
        ssh(host, "cd " + shlex.quote(release) + " && " + shlex.join(command))
        return
    destination = ROOT / "data" / "raw" / "jetson" / (("qp_" if qp else "") + stamp())
    destination.mkdir(parents=True, exist_ok=False)
    # A fresh local directory preserves prior downloads. Validate archive members
    # before extraction; do not permit links or paths outside this destination.
    with tempfile.TemporaryFile() as archive:
        ssh(host, "cd " + shlex.quote(release) + " && tar -czf - data/raw", stdout=archive)
        archive.seek(0)
        with tarfile.open(fileobj=archive, mode="r:gz") as bundle:
            for member in bundle.getmembers():
                relative = PurePosixPath(member.name)
                resolved = (destination / member.name).resolve()
                if (relative.is_absolute() or ".." in relative.parts
                        or "\\" in member.name or ":" in member.name
                        or not resolved.is_relative_to(destination.resolve())
                        or relative.parts[:2] != ("data", "raw")
                        or not (member.isfile() or member.isdir())):
                    raise RuntimeError("Unexpected archive member: " + member.name)
            bundle.extractall(destination)
    manifest = {}
    for path in sorted(destination.rglob("*")):
        if path.is_file():
            manifest[path.relative_to(destination).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not manifest:
        raise SystemExit("No data files downloaded")
    # Compare with hashes calculated on the Jetson to verify the complete transfer.
    hash_code = "import pathlib,hashlib,json; print(json.dumps({p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in pathlib.Path('data/raw').rglob('*') if p.is_file()}))"
    result = ssh(host, "cd " + shlex.quote(release) + " && python3 -c " + shlex.quote(hash_code),
                 capture_output=True, text=True)
    remote_hashes = json.loads(result.stdout)
    if manifest != remote_hashes:
        raise RuntimeError("Remote/local hashes differ; collection may still be running")
    (destination / "transfer.json").write_text(json.dumps({
        "source_host": host, "source_release": release, "sha256": manifest,
        "verified": True}, indent=2), encoding="utf-8")
    print("Downloaded and SHA-256 verified:", destination)


if __name__ == "__main__":
    main()
