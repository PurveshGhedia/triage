"""Serve Agent 1 and/or Agent 2 as A2A services (Google ADK + A2A protocol).

Usage (run as a module from the repo root):
    python -m scripts.serve_agents --agent both                     # local: 8101 + 8102
    python -m scripts.serve_agents --agent perception --port 8101
    python -m scripts.serve_agents --agent planner --port 8102
    python -m scripts.serve_agents --agent perception --public-url https://agent1-xyz.run.app

On Cloud Run each agent is its own service: pass --agent plus the service's public
URL, because the agent card advertises that address and every client call goes to it.
The port defaults to $PORT, which Cloud Run sets.
"""
from __future__ import annotations

import argparse
import os
import threading
import warnings

warnings.filterwarnings("ignore", message=r"\[EXPERIMENTAL\]")

import uvicorn

from agents.orchestrator.a2a_services import (
    build_perception_app,
    build_planner_app,
    split_public_url,
)
from agents.orchestrator.ports import InProcessPerception, InProcessPlanner
from shared.agent_factory import build_perception_agent, build_second_opinion_agent


def make_perception(use_llm: bool, concurrency: int, tools: str = "mock") -> InProcessPerception:
    return InProcessPerception(
        agent=build_perception_agent(use_llm, tools_mode=tools),
        reinspection_agent=build_second_opinion_agent(use_llm, tools_mode=tools),
        max_workers=concurrency,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=["perception", "planner", "both"], required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8101)))
    parser.add_argument("--public-url", default=None,
                        help="Address the agent card advertises (Cloud Run service URL).")
    parser.add_argument("--use-llm-selector", action="store_true")
    parser.add_argument("--tools", choices=["mock", "vision"], default="mock")
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    def advertised(port: int) -> tuple[str, str, int]:
        if args.public_url:
            return split_public_url(args.public_url)
        return "http", "localhost", port

    servers = []
    if args.agent in ("perception", "both"):
        port = args.port
        protocol, host, adv_port = advertised(port)
        app = build_perception_app(make_perception(args.use_llm_selector, args.concurrency, args.tools),
                                   host, adv_port, protocol)
        servers.append((app, port, "perception"))
    if args.agent in ("planner", "both"):
        port = args.port + 1 if args.agent == "both" else args.port
        protocol, host, adv_port = advertised(port)
        app = build_planner_app(InProcessPlanner(), host, adv_port, protocol)
        servers.append((app, port, "planner"))

    for app, port, name in servers:
        print(f"{name} agent: http://localhost:{port}/.well-known/agent-card.json")

    if len(servers) == 1:
        app, port, _ = servers[0]
        uvicorn.run(app, host=args.host, port=port, log_level="info")
        return
    threads = []
    for app, port, _ in servers[1:]:
        server = uvicorn.Server(uvicorn.Config(app, host=args.host, port=port, log_level="info"))
        t = threading.Thread(target=server.run, daemon=True)
        t.start()
        threads.append(t)
    app, port, _ = servers[0]
    uvicorn.run(app, host=args.host, port=port, log_level="info")


if __name__ == "__main__":
    main()
