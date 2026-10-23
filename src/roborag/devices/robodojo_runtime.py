"""Own the native simulator lifecycle for the MCP service.

Configuration/layout setup follows the pinned MIT RoboICL robodojo adapter.
No upstream source or native task/scoring implementation is modified.
"""
import argparse
from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path


class NoPolicyClient:
    def __init__(self, *args, **kwargs): pass
    def call(self, *, func_name, **kwargs):
        if func_name != "reset":
            raise RuntimeError("RoboDojo control is owned by the MCP service")
    def close(self): pass


@contextmanager
def open_environment(*, register_task=True):
    from roboicl.paths import ROBODOJO, configure_imports
    configure_imports("sim")
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--layout", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    assets = args.data_root.resolve() / "Assets"
    layout = assets / "Eval_Layout/RoboDojo/arx_x5" / str(args.seed) / f"{args.task}_{args.layout}.json"
    if not layout.is_file():
        raise FileNotFoundError(layout)
    os.environ.update(ROBOICL_DATA_ROOT=str(args.data_root.resolve()), ROBOICL_RESULTS_ROOT=str(output), ROBODOJO_RUN_ID=output.name)
    os.chdir(output)
    global_config = importlib.import_module("env.global_configs")
    global_config.ASSETS_PATH = str(assets)
    global_config.OBJECTS_PATH = str(assets / "Object/RoboDojo")
    global_config.ROBOTS_PATH = str(assets / "Robots")
    app = AppLauncher(args).app
    env = None
    try:
        from omegaconf import OmegaConf
        from utils.load_file import load_yaml
        from utils.pipeline_utils import process_config, process_randomization
        from src.eval_client.ik_continuity import IKContinuityGuard
        module = importlib.import_module("src.eval_client.eval_env")
        module.WsModelClient = NoPolicyClient
        root = ROBODOJO / "env_cfg"
        base = load_yaml(root / "arx_x5.yml")
        base.update(task_name=args.task, num_envs=1, device_id=0, eval_batch=False,
                    policy_name="RoboICL", additional_info="mcp-cap", seed=args.seed,
                    physx_monitor_enabled=False, eval_num=1)
        registry = importlib.import_module("task.RoboDojo.task_registry")
        task_config = registry.task_config_path(ROBODOJO / "task/RoboDojo/config", args.task)
        config = OmegaConf.create({
            **{key: load_yaml(root / key / f"{base['config'][key]}.yml") for key in ("sim", "scene", "camera", "robot")},
            "task_env": load_yaml(task_config), "eval_cfg": base,
            "deploy_cfg": {"policy_name": "RoboICL", "port": 1, "host": "127.0.0.1", "protocol": "ws",
                           "evaluation_id": output.name, "trial_id": output.name, "action_case_id": args.task},
        })
        config = process_randomization(config)
        config, _ = process_config(config, task_name=args.task)
        config.sim.scene.num_envs = 1
        config.sim.seed = [0]
        config.camera.default_frequency = base["observation"].get("collect_freq", 0)
        env = module.create_eval_env(config, app)
        env.seed_manager.seed_info = {args.layout: {"scene_layout": str(layout)}}
        env.seed_manager.seed_list = [args.layout]
        env.seed_manager.st_idx, env.seed_manager.ed_idx, env.seed_manager.idx = 0, 1, 0
        env.env_seeds = [args.layout]
        env.reset(seed=[args.layout])
        # EvalEnv.run_eval normally registers these before dispatching a policy.
        # Without registration an empty reward-check list can end the episode.
        if register_task:
            env.run_reward()
            if hasattr(env, "get_score"):
                env.get_score()
            if getattr(env, "interact", False) and hasattr(env, "query_support_arm_traj"):
                for index in env.get_running_env_idx_list():
                    env.query_support_arm_traj(env_idx=index)
        env.ik_continuity_guard = IKContinuityGuard(0.15, log_path=output / "ik-continuity.jsonl")
        from roboicl.rollout import _ready_observation
        if register_task:
            _ready_observation(env)
        (output / "ready.json").write_text(json.dumps({"task": args.task, "layout": args.layout, "seed": args.seed,
            "backend": "native RoboDojo / Isaac Sim",
            "task_registration": "runtime" if register_task else "caller",
            "benchmark_evaluation": False if register_task else None}))
        yield env, output
    finally:
        try:
            if env is not None:
                env.model_client.close()
                env.close()
        finally:
            app.close()
