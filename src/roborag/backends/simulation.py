"""Native RoboDojo episode for GPT-generated plans and MCP action chunks."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import time
import traceback
import types
from pathlib import Path

from roboicl.paths import ROBODOJO, configure_imports

configure_imports("sim")
from isaaclab.app import AppLauncher


class OfflineClient:
    def __init__(self, *args, **kwargs):
        pass

    def call(self, **kwargs):
        return None

    def close(self):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hexashell", type=Path, required=True)
    parser.add_argument("--layout", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task", default="stack_blocks")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--agent-workspace", type=Path)
    parser.add_argument("--max-wall-seconds", type=float, default=7200)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    output = args.output.resolve()
    for name in ("inbox", "outbox"):
        (output / name).mkdir(parents=True, exist_ok=True)
    exchange = args.agent_workspace.resolve() if args.agent_workspace else output
    os.chdir(output)
    assets = Path(os.environ["ROBOICL_DATA_ROOT"]) / "Assets"
    global_cfg = importlib.import_module("env.global_configs")
    global_cfg.ASSETS_PATH = str(assets)
    global_cfg.OBJECTS_PATH = str(assets / "Object/RoboDojo")
    global_cfg.ROBOTS_PATH = str(assets / "Robots")
    app = AppLauncher(args).app
    from roborag.backends.robodojo import EpisodeAPI, write_json
    from roborag.capabilities import measurement_for
    from roborag.control.execution import execute_delta_actions
    from roborag.control.homing import execute_home
    from roborag.tasks import task_spec

    class LiveAPI(EpisodeAPI):
        def state(self):
            value = super().state()
            value["action_state"] = {}
            for side, arm in self.arms.items():
                value["action_state"][f"{side}_ee_pose"] = self.geometry.matrix_to_pose(arm.ee())
                value["action_state"][f"{side}_ee_joint_state"] = [arm.gripper_target]
            return value

    env = None
    api = None
    try:
        from omegaconf import OmegaConf
        from utils.load_file import load_yaml
        from utils.pipeline_utils import process_config, process_randomization

        module = importlib.import_module("src.eval_client.eval_env")
        module.WsModelClient = OfflineClient
        task = args.task
        spec = task_spec(task.removesuffix("_random"))
        base = load_yaml(ROBODOJO / "env_cfg/arx_x5.yml")
        base.update(
            task_name=task,
            num_envs=1,
            device_id=0,
            eval_batch=False,
            policy_name="RoboRAGMCP",
            additional_info="gpt_plan_chunk",
            seed=args.seed,
            physx_monitor_enabled=False,
            eval_num=1,
        )
        registry = importlib.import_module("task.RoboDojo.task_registry")
        config = OmegaConf.create(
            {
                "sim": load_yaml(ROBODOJO / "env_cfg/sim/sim_config.yml"),
                "scene": load_yaml(ROBODOJO / "env_cfg/scene/default.yml"),
                "robot": load_yaml(ROBODOJO / "env_cfg/robot/dual_x5.yml"),
                "camera": load_yaml(ROBODOJO / "env_cfg/camera/camera_config.yml"),
                "task_env": load_yaml(
                    registry.task_config_path(ROBODOJO / "task/RoboDojo/config", task)
                ),
                "eval_cfg": base,
                "deploy_cfg": dict(
                    policy_name="RoboRAGMCP", port=1, host="127.0.0.1", protocol="ws"
                ),
            }
        )
        config = process_randomization(config)
        config, _ = process_config(config, task_name=task)
        config.sim.scene.num_envs = 1
        config.sim.seed = [0]
        config.camera.default_frequency = 25
        env = module.create_eval_env(config, app)
        layout = assets / f"Eval_Layout/RoboDojo/arx_x5/{args.seed}/{task}_{args.layout}.json"
        env.seed_manager.seed_info = {args.layout: {"scene_layout": str(layout)}}
        env.seed_manager.seed_list = [args.layout]
        env.seed_manager.st_idx, env.seed_manager.ed_idx, env.seed_manager.idx = 0, 1, 0
        env.env_seeds = [args.layout]
        env.reset(seed=[args.layout])
        rejection_errors = ()
        if spec.ik_joint_increment_rad is not None:
            from src.eval_client.ik_continuity import IKContinuityError, IKContinuityGuard

            # Uses only robot joints and the native solver's proposed targets.
            # The native guard rejects both arms/grippers before queue mutation.
            env.ik_continuity_guard = IKContinuityGuard(
                max_joint_increment_rad=spec.ik_joint_increment_rad,
                log_path=output / "ik-continuity.jsonl",
            )
            rejection_errors = (IKContinuityError,)
        api = LiveAPI(
            env,
            output,
            task_instruction=spec.public_instruction,
            support_robots=spec.support_robots,
        )

        def publish_ready():
            # Must run after native run_eval registers its reward/score checks.
            # An empty reward registry after reset is not a task verdict.
            api.save_observation(output / "obs")
            if exchange != output:
                api.save_observation(exchange / "obs")
            api.save_observation(output / "observations/000_initial")
            write_json(
                output / "ready.json",
                dict(
                    step=api.step,
                    step_limit=int(env.step_lim),
                    task=task,
                    layout=args.layout,
                    seed=args.seed,
                ),
            )

        start = time.monotonic()

        def episode(instance):
            publish_ready()
            if args.probe_only:
                write_json(
                    output / "initialization-probe.json",
                    dict(
                        role="infrastructure_probe",
                        scored=False,
                        task=task,
                        state=api.state(),
                        policy_action_steps=api.step,
                        controlled_arms=[arm._robot.arm_name for arm in api.arms.values()],
                        native_robot_count=len(env.robot_manager.robot_list),
                    ),
                )
                api.stop("initialization_probe")
                return
            processed = set()
            counter = 0
            while not api.over:
                if time.monotonic() - start > args.max_wall_seconds:
                    raise TimeoutError("Wall-time limit; episode incomplete, no valid score")
                requests = sorted((exchange / "inbox").glob("*.json"))
                pending = [p for p in requests if p.name not in processed]
                if not pending:
                    time.sleep(0.05)
                    continue
                path = pending[0]
                request = json.loads(path.read_text())
                if request.get("abort"):
                    raise RuntimeError("Policy runner aborted; episode incomplete")
                before = api.step
                started = time.monotonic()
                if request.get("finish") == "agent_exit":
                    api.stop("agent_exit")
                    result, code = dict(episode_over=True, step=api.step, reason="agent_exit"), 0
                elif measurement := measurement_for(request, spec.capabilities):
                    result, code = measurement.measure(
                        api, args.hexashell, request[measurement.request_key]
                    )
                else:
                    result = (
                        execute_delta_actions(
                            api, request["actions"], rejection_errors=rejection_errors
                        )
                        if "actions" in request
                        else execute_home(api, request["chunk"])
                    )
                    code = 0
                counter += 1
                api.save_observation(output / "obs")
                if exchange != output:
                    api.save_observation(exchange / "obs")
                snapshot = output / f"observations/{counter:03d}_step{api.step:03d}"
                api.save_observation(snapshot)
                receipt = dict(
                    result=result,
                    exit_code=code,
                    step_before=before,
                    step_after=api.step,
                    wall_seconds=time.monotonic() - started,
                    observation=str(snapshot),
                )
                with (output / "commands.jsonl").open("a") as log:
                    log.write(json.dumps(dict(request=request, **receipt)) + "\n")
                write_json(
                    output / "progress.json",
                    dict(subgoal=request.get("subgoal"), counter=counter, **receipt),
                )
                if exchange != output:
                    write_json(
                        exchange / "outbox" / path.name,
                        {k: v for k, v in receipt.items() if k != "observation"},
                    )
                write_json(output / "outbox" / path.name, receipt)
                processed.add(path.name)
            write_json(
                output / "episode_end.json",
                dict(
                    step=api.step,
                    commands=api.commands_used,
                    reason=api.stop_reason or "native_environment",
                ),
            )

        env.eval_one_episode = types.MethodType(episode, env)
        env.run_eval()
        write_json(
            output / "sim_complete.json",
            dict(complete=True, native_result=str(Path(env.save_dir) / "_result.json")),
        )
    except BaseException as exc:
        write_json(
            output / "sim_failure.json",
            dict(error=type(exc).__name__, detail=str(exc), complete=False),
        )
        traceback.print_exc()
        # Preserve incomplete video evidence without manufacturing a score.
        if env is not None:
            try:
                env.save_video(0, str(output / "incomplete.mp4"), "interrupted")
            except Exception:
                pass
        raise
    finally:
        if api is not None:
            api.close()
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
