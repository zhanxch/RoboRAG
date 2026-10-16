"""WebSocket entry point for RoboRAG and matched control baselines."""

import argparse
import asyncio
import os

from roboicl.paths import configure_imports

configure_imports()
from client_server.ws.model_server import PolicyServer, PolicyServerConfig
from roboicl.policy.dialogue_policy import Model as BaseModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reasoning-effort", default="medium")
    args = parser.parse_args()
    model_type = BaseModel
    if os.environ.get("ROBORAG_MODE", "zero") != "zero":
        from .robot_memory import RetrievalModel
        model_type = RetrievalModel
    model = model_type({"policy_name": "RoboRAG", "model": "gpt-6-astra", "bench_name": "RoboDojo",
                        "task_name": args.task, "seed": args.seed, "action_type": "ee", "max_chunk": 64,
                        "max_tool_rounds": 16, "reasoning_effort": args.reasoning_effort})
    server = PolicyServer(model, PolicyServerConfig(host="127.0.0.1", port=args.port,
                                                   ws_ping_interval_s=30, ws_ping_timeout_s=600))
    asyncio.run(server.serve_forever())


if __name__ == "__main__":
    main()
