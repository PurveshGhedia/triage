"""Start Agent 1 and Agent 2 as real A2A servers on localhost, in background threads.

Used by `run_e2e --transport a2a --spawn-local` and by the A2A tests. Each server
listens on its own port and is reached over HTTP exactly as it would be on Cloud
Run, so this exercises the genuine A2A path without any deployment.
"""
from __future__ import annotations

import socket
import threading
import time
from dataclasses import dataclass

import uvicorn

from agents.orchestrator.a2a_services import build_perception_app, build_planner_app
from agents.orchestrator.ports import InProcessPerception, InProcessPlanner


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _serve(app, port: int) -> uvicorn.Server:
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    deadline = time.time() + 15
    while time.time() < deadline:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            return server
        except OSError:
            time.sleep(0.05)
    raise RuntimeError(f"A2A server on port {port} did not start")


@dataclass
class LocalServices:
    perception_url: str
    planner_url: str
    _servers: list

    def stop(self) -> None:
        for server in self._servers:
            server.should_exit = True


def start_local_services(
    perception: InProcessPerception, planner: InProcessPlanner | None = None
) -> LocalServices:
    p1, p2 = free_port(), free_port()
    s1 = _serve(build_perception_app(perception, "127.0.0.1", p1), p1)
    s2 = _serve(build_planner_app(planner or InProcessPlanner(), "127.0.0.1", p2), p2)
    return LocalServices(f"http://127.0.0.1:{p1}", f"http://127.0.0.1:{p2}", [s1, s2])
