#!/usr/bin/env python3
"""Unscored renderer calibration check; no robot policy or benchmark truth feed."""
import argparse
import json
from pathlib import Path
import traceback

from isaacsim import SimulationApp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    app = SimulationApp({"headless": True, "width": 640, "height": 480})
    try:
        import numpy as np
        from PIL import Image
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import VisualSphere
        from isaacsim.core.utils.stage import get_current_stage
        from isaacsim.sensors.camera import Camera
        from pxr import UsdLux

        world = World(stage_units_in_meters=1.0, physics_dt=.004, rendering_dt=.04)
        light = UsdLux.DomeLight.Define(get_current_stage(), "/World/Light")
        light.CreateIntensityAttr(1000.)
        positions = [[0., .2, 0.], [.2, -.2, 0.], [-.2, -.15, 0.]]
        for index, position in enumerate(positions):
            color = np.eye(3)[index]
            world.scene.add(VisualSphere(f"/World/Marker{index}", name=f"marker{index}",
                position=np.array(position), radius=.015, color=color))
        # Static marker projection does not depend on capture cadence. Use the
        # standalone application's native render cadence to avoid its 60 Hz
        # app-loop / 25 Hz benchmark frequency configuration mismatch.
        camera = Camera("/World/Camera", resolution=(640, 480))
        camera.set_lens_distortion_model("pinhole")
        camera.set_focal_length(10.)
        camera.set_horizontal_aperture(22.212, False)
        camera.set_vertical_aperture(14.266, False)
        camera.set_clipping_range(.005, 10.)
        camera.set_world_pose(np.array([0., 0., 1.]), np.array([1., 0., 0., 0.]), camera_axes="usd")
        world.reset()
        camera.initialize()
        for _ in range(30):
            world.step(render=True)
        rgb = np.asarray(camera.get_rgba())[:, :, :3].astype(np.uint8)
        Image.fromarray(rgb).save(args.output / "markers.png")
        sdk = np.asarray(camera.get_intrinsics_matrix(device="cpu"), float)
        square = sdk.copy()
        square[1, 1] = square[0, 0]
        rows = []
        for index, position in enumerate(positions):
            pixels = rgb.astype(float)
            others = np.max(np.delete(pixels, index, axis=2), axis=2)
            mask = (pixels[:, :, index] > others * 1.5 + 30) & (pixels[:, :, index] > 80)
            y, x = np.nonzero(mask)
            if len(x) < 10:
                raise RuntimeError(f"Marker {index} not reliably visible: {len(x)} pixels")
            observed = np.array([x.mean(), y.mean()])
            # Identity USD camera at z=1: ROS camera point is [x,-y,1-z].
            point = np.array([position[0], -position[1], 1-position[2]])
            predictions = {}
            for name, K in [("sdk", sdk), ("square_pixels", square)]:
                prediction = K @ point
                prediction = prediction[:2] / prediction[2]
                predictions[name] = dict(pixel=prediction.tolist(), error_pixels=float(np.linalg.norm(prediction-observed)))
            rows.append(dict(marker=index, world_position=position, detected_pixels=len(x), observed_pixel=observed.tolist(), predictions=predictions))
        result = dict(scored=False, policy_invoked=False, setup="Independent synthetic scene with benchmark head-camera aperture/resolution settings",
            sdk_intrinsics=sdk.tolist(), square_pixel_intrinsics=square.tolist(), markers=rows,
            mean_sdk_error_pixels=float(np.mean([r['predictions']['sdk']['error_pixels'] for r in rows])),
            mean_square_pixel_error_pixels=float(np.mean([r['predictions']['square_pixels']['error_pixels'] for r in rows])))
        (args.output / "calibration.json").write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps(result), flush=True)
    except BaseException as exc:
        (args.output / "failure.json").write_text(json.dumps(dict(error=type(exc).__name__, detail=str(exc), traceback=traceback.format_exc()), indent=2)+"\n")
        traceback.print_exc()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
