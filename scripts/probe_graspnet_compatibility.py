#!/usr/bin/env python3
"""Read-only GraspNet preflight; no neural inference or physical rollout.

Execute the checked-out RoboDojo get_obs implementation with synthetic sensor
fixtures, inspect one real HDF5, and probe the selected Python environment.
Run with the policy interpreter, not Isaac Sim's interpreter.
"""
import argparse
import ast
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]


def source_record(path):
    return {"path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def module_probe(name):
    try:
        module = importlib.import_module(name)
        return {"import_ok": True, "version": str(getattr(module, "__version__", "unknown"))}
    except Exception as exc:
        return {"import_ok": False, "error": f"{type(exc).__name__}: {exc}"}


def observation_contract(source):
    """Run actual get_obs source; synthetic camera/robot-free fixtures only."""
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ObsManager")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "get_obs")
    namespace = {"np": np, "deepcopy": deepcopy}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    raw = np.array([[[0.25], [0.50], [1.00]], [[0.75], [1.25], [2.00]]], dtype=np.float32)
    k = np.array([[100., 0., 1.5], [0., 100., 1.], [0., 0., 1.]])
    transform = np.eye(4)
    results = []
    for label, depth_annotator, export in [
        ("rgb_only_default", False, False),
        ("export_flags_without_depth_annotator", False, True),
        ("depth_annotator_and_export_flags", True, True),
    ]:
        capture = {"rgb": [{"data": np.zeros((2, 3, 4), dtype=np.uint8)}]}
        if depth_annotator:
            capture["distance_to_image_plane"] = [{"data": raw.copy()}]
        obj = SimpleNamespace(
            num_envs=1, collect_freq=25, instruction=["synthetic contract fixture"],
            camera_manager=SimpleNamespace(num_cams=1, camera_names=[["cam_head"]],
                get_camera_intrinsics=lambda *a: k.copy(),
                get_camera_extrinsics=lambda *a: transform.copy()),
            capture_manager=SimpleNamespace(step=lambda **kw: {0: capture}),
            robot_manager=None, collect_shape=True, collect_approximate_depth=False,
            collect_depth=export, collect_intrinsic_matrix=export,
            collect_extrinsic_matrix=export,
            ANNOTATORS_TO_COLLECT={"rgb": "color", "depth": "depth",
                                   "distance_to_image_plane": "distance_to_image_plane"},
        )
        camera = namespace["get_obs"](obj)[0]["vision"]["cam_head"]
        expected = {"color", "shape"}
        if export:
            expected |= {"intrinsic_matrix", "extrinsic_matrix"}
        if depth_annotator and export:
            expected.add("depth")
        passed = set(camera) == expected and camera["color"].shape == (2, 3, 3)
        if "depth" in camera:
            passed = passed and camera["depth"].dtype == np.float32
            passed = passed and np.array_equal(camera["depth"], raw[..., 0])
        results.append({"case": label, "passed": bool(passed), "fields": sorted(camera),
                        "depth_present": "depth" in camera})
    return {"kind": "actual_get_obs_method_with_synthetic_sensor_fixtures",
            "real_rendering_tested": False, "cases": results,
            "all_passed": all(row["passed"] for row in results)}


def hdf5_inventory(path):
    with h5py.File(path, "r") as data:
        cameras = {}
        for name, camera in data["vision"].items():
            cameras[name] = {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                             for key, value in camera.items() if isinstance(value, h5py.Dataset)}
        return {"path": str(path), "frequency_hz": int(data["additional_info/frequency"][()]),
                "cameras": cameras,
                "depth_available": any("depth" in k for c in cameras.values() for k in c)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robodojo", type=Path, default=ROOT / "vendor/RoboICL/third_party/RoboDojo")
    parser.add_argument("--hdf5", type=Path,
                        default=ROOT / "data/runtime-data/stack_blocks/train/episode_0000000.hdf5")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = [args.robodojo / p for p in ["env_cfg/arx_x5.yml", "env_cfg/camera/camera_config.yml",
        "env/observation_manager/obs_manager.py", "env/camera_manager/camera_manager.py"]]
    config = yaml.safe_load(files[0].read_text())
    camera_config = yaml.safe_load(files[1].read_text())
    modules = {name: module_probe(name) for name in
               ["torch", "numpy", "open3d", "graspnetAPI", "pointnet2_utils", "knn_pytorch", "MinkowskiEngine", "gsnet"]}
    torch = importlib.import_module("torch") if modules["torch"]["import_ok"] else None
    report = {
        "schema": "graspnet.robodojo.preflight.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Installed-environment and source-contract preflight; NOT a grasp-success or model-latency benchmark.",
        "python": {"executable": sys.executable, "version": sys.version.split()[0]},
        "runtime": {"modules": modules, "nvcc_on_path": shutil.which("nvcc"),
                    "torch_cuda": torch.version.cuda if torch else None,
                    "gpu": torch.cuda.get_device_name(0) if torch and torch.cuda.is_available() else None},
        "robodojo": {
            "revision": subprocess.check_output(["git", "-C", str(args.robodojo), "rev-parse", "HEAD"], text=True).strip(),
            "files": [source_record(p) for p in files],
            "default_observation": config["observation"],
            "configured_annotators": camera_config["annotator"],
        },
        "observation_contract": observation_contract(files[2]),
        "recorded_sample": hdf5_inventory(args.hdf5),
        "neural_inference_executed": False,
        "physical_trials": 0,
        "model_latency_ms": None,
        "grasp_success_rate": None,
        "notes": [
            "graspnetAPI is a dataset/pose/evaluation toolbox; an independent detector and checkpoint are required.",
            "Import failures apply to this interpreter and its sys.path, not all possible environments.",
            "nvcc_on_path=null means unavailable on PATH; it is not a full filesystem CUDA toolkit audit.",
            "Only the named recorded HDF5 is inspected; no inference about all dataset episodes.",
            "Synthetic contract tests validate field plumbing, not rendered depth, camera calibration, grasp geometry, IK, or physics.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"report": str(args.output), "observation_contract_passed": report["observation_contract"]["all_passed"],
                      "missing_or_failed_modules": [k for k, v in modules.items() if not v["import_ok"]],
                      "recorded_depth_available": report["recorded_sample"]["depth_available"],
                      "model_latency_ms": None, "physical_trials": 0}, ensure_ascii=False, indent=2))
    return 0 if report["observation_contract"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
