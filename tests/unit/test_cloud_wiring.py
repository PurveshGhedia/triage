"""What the Cloud Run deployment relies on: ID-token auth between private services, per-request agent
options, environment-driven remote agents, and a guard against stale demo folders."""
from __future__ import annotations

import base64
import json
import struct
import threading
import time
import zlib

import httpx
import pytest

pytest.importorskip("google.adk")

from agents.orchestrator.a2a_auth import IdTokenAuth  # noqa: E402
from agents.orchestrator.a2a_client import A2APerception, A2APlanner  # noqa: E402
from agents.orchestrator.a2a_services import build_perception_app, build_planner_app  # noqa: E402
from agents.orchestrator.local_services import _serve, free_port  # noqa: E402
from agents.orchestrator.models import TileSpec  # noqa: E402
from agents.orchestrator.ports import InProcessPerception, InProcessPlanner  # noqa: E402
from agents.orchestrator.runner import RunOptions, execute_run  # noqa: E402
from agents.perception.agent import PerceptionAgent  # noqa: E402
from shared.tool_interface import ToolClient, ToolResult  # noqa: E402


def jwt(exp: float) -> str:
    seg = lambda d: base64.urlsafe_b64encode(json.dumps(d).encode()).decode().rstrip("=")
    return f"{seg({'alg': 'none'})}.{seg({'exp': exp})}.sig"


# --- IdTokenAuth -------------------------------------------------------------------------------


def test_auth_adds_a_bearer_token_for_the_service_url_audience():
    seen = []
    auth = IdTokenAuth("https://agent1-abc.run.app/", token_fn=lambda aud: seen.append(aud) or "tok-1")
    seen_headers = []
    transport = httpx.MockTransport(lambda req: seen_headers.append(req.headers["authorization"]) or httpx.Response(200))
    with httpx.Client(auth=auth, transport=transport) as c:
        c.get("https://agent1-abc.run.app/x")
    assert seen == ["https://agent1-abc.run.app"] and seen_headers == ["Bearer tok-1"]       # trailing slash stripped


def test_auth_works_for_async_clients_too():
    import asyncio

    auth = IdTokenAuth("https://a.run.app", token_fn=lambda aud: "tok-async")
    got = []

    async def go():
        t = httpx.MockTransport(lambda req: got.append(req.headers["authorization"]) or httpx.Response(200))
        async with httpx.AsyncClient(auth=auth, transport=t) as c:
            await c.get("https://a.run.app/")

    asyncio.run(go())
    assert got == ["Bearer tok-async"]


def test_token_is_cached_then_refreshed_before_it_expires():
    now = [1000.0]
    calls = []

    def fetch(aud):
        calls.append(aud)
        return jwt(now[0] + 3600)                                  # valid for an hour

    auth = IdTokenAuth("https://a.run.app", token_fn=fetch, clock=lambda: now[0])
    first = auth.token()
    now[0] += 1800
    assert auth.token() == first and len(calls) == 1               # still fresh: no second fetch
    now[0] += 1500                                                 # 55 min in: inside the 5-minute margin
    auth.token()
    assert len(calls) == 2


def test_a_token_with_no_readable_expiry_gets_a_safe_default_lifetime():
    now = [0.0]
    calls = []
    auth = IdTokenAuth("https://a.run.app", token_fn=lambda a: calls.append(1) or "opaque", clock=lambda: now[0])
    auth.token(); now[0] = 40 * 60; auth.token()
    assert len(calls) == 1                                         # fine at 40 min
    now[0] = 46 * 60; auth.token()
    assert len(calls) == 2                                         # refreshed once inside 5 min of the 50 min default


def test_token_failures_propagate_so_the_run_falls_back():
    def boom(aud):
        raise RuntimeError("no metadata server")

    with pytest.raises(RuntimeError, match="no metadata server"):
        IdTokenAuth("https://a.run.app", token_fn=boom).token()


# --- a private A2A service: header required on the card fetch AND the call -------------------------


class RequireBearer:
    """ASGI wrapper that rejects any request without the right token, like Cloud Run IAM does."""

    def __init__(self, app, token):
        self.app, self.token, self.log = app, token, []

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = dict(scope["headers"])
            ok = headers.get(b"authorization") == f"Bearer {self.token}".encode()
            self.log.append((scope["path"], ok))
            if not ok:
                await send({"type": "http.response.start", "status": 403, "headers": []})
                await send({"type": "http.response.body", "body": b"forbidden"})
                return
        await self.app(scope, receive, send)


class Fixed(ToolClient):
    name = "classifier"

    def run(self, tile_path):
        return ToolResult(self.name, 0.9, "minor-damage")


def specs(n=3):
    return [TileSpec(tile_id=f"t{i}", tile_path=f"/x/t{i}.png", latitude=10 + i * 0.001, longitude=20.0) for i in range(n)]


def test_a_private_service_rejects_unauthenticated_callers_and_accepts_the_token():
    perception = InProcessPerception(PerceptionAgent(tools=[Fixed()], sleep=lambda s: None), max_workers=2)
    port = free_port()
    guarded = RequireBearer(build_perception_app(perception, "127.0.0.1", port), "secret-id-token")
    server = _serve(guarded, port)
    url = f"http://127.0.0.1:{port}"
    try:
        with pytest.raises(RuntimeError):
            A2APerception(url).assess("A", specs())                                    # no credentials: refused
        audiences = []
        auth = IdTokenAuth(url, token_fn=lambda aud: audiences.append(aud) or "secret-id-token")
        results = A2APerception(url, auth=auth).assess("A", specs())
        assert len(results) == 3 and all(r.damage_severity == "minor-damage" for r in results)
        assert audiences and set(audiences) == {url}                                   # token minted for THIS service
        authed_paths = [p for p, ok in guarded.log if ok]
        assert any("agent-card" in p for p in authed_paths) and len(authed_paths) >= 2  # card fetch and call both carried it
    finally:
        server.should_exit = True


def test_the_planner_service_can_be_private_too():
    port = free_port()
    guarded = RequireBearer(build_planner_app(InProcessPlanner(), "127.0.0.1", port), "tok")
    server = _serve(guarded, port)
    url = f"http://127.0.0.1:{port}"
    try:
        with pytest.raises(RuntimeError):
            A2APlanner(url).plan("A", [], 1, 500.0)
        out = A2APlanner(url, auth=IdTokenAuth(url, token_fn=lambda a: "tok")).plan("A", [], 1, 500.0)
        assert out.rankings == []
    finally:
        server.should_exit = True


# --- per-request options reach the remote Agent 1 ------------------------------------------------------


def test_the_remote_agent_is_built_from_each_requests_options_and_cached_per_option_set():
    built = []

    def factory(tools="mock", use_llm_selector=False, vision_variant=None):
        built.append((tools, use_llm_selector, vision_variant))
        return InProcessPerception(PerceptionAgent(tools=[Fixed()], sleep=lambda s: None), max_workers=1)

    port = free_port()
    server = _serve(build_perception_app(None, "127.0.0.1", port, perception_factory=factory), port)
    url = f"http://127.0.0.1:{port}"
    try:
        A2APerception(url).assess("A", specs(1))                                                       # no options
        A2APerception(url, options={"tools": "vision", "use_llm_selector": True, "vision_variant": "fractions"}).assess("A", specs(1))
        A2APerception(url, options={"tools": "vision", "use_llm_selector": True, "vision_variant": "fractions"}).assess("A", specs(1))
        assert built == [("mock", False, None), ("vision", True, "fractions")]                          # second vision call reused
    finally:
        server.should_exit = True


def test_a_fixed_agent_is_used_when_no_factory_is_configured():
    perception = InProcessPerception(PerceptionAgent(tools=[Fixed()], sleep=lambda s: None), max_workers=1)
    port = free_port()
    server = _serve(build_perception_app(perception, "127.0.0.1", port), port)
    try:
        out = A2APerception(f"http://127.0.0.1:{port}", options={"tools": "vision"}).assess("A", specs(1))
        assert out[0].damage_severity == "minor-damage"
    finally:
        server.should_exit = True


# --- dashboard / runner wiring ---------------------------------------------------------------------------


def make_png() -> bytes:
    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"\x00\x80\x80\x80")) + chunk(b"IEND", b""))


@pytest.fixture
def demo(tmp_path):
    d = tmp_path / "demo" / "storm"
    (d / "images").mkdir(parents=True)
    geo = {}
    for i in range(5):
        tile = f"storm_{i:08d}_post_disaster"
        (d / "images" / f"{tile}.png").write_bytes(make_png())
        geo[tile] = {"latitude": 10 + (i // 3) * 0.004, "longitude": 20 + (i % 3) * 0.004}
    (d / "geo.json").write_text(json.dumps(geo))
    (d / "demo.json").write_text(json.dumps({"adjacency_radius_m": 700, "tile_count": 5}))
    return tmp_path / "demo"


def events_of(tmp_path, result):
    return [json.loads(l) for l in (tmp_path / "runs" / result.run_id / "events.jsonl").read_text().splitlines()]


def test_runs_use_the_remote_agent_urls_when_given_and_apply_the_options(demo, tmp_path):
    seen = []

    def factory(tools="mock", use_llm_selector=False, vision_variant=None):
        seen.append((tools, use_llm_selector, vision_variant))
        return InProcessPerception(PerceptionAgent(tools=[Fixed()], sleep=lambda s: None), max_workers=2)

    p1, p2 = free_port(), free_port()
    s1 = _serve(build_perception_app(None, "127.0.0.1", p1, perception_factory=factory), p1)
    s2 = _serve(build_planner_app(InProcessPlanner(), "127.0.0.1", p2), p2)
    try:
        result = execute_run(RunOptions(
            disaster="storm", demo_root=demo, offline=True, transport="a2a", rounds=0,
            agent1_url=f"http://127.0.0.1:{p1}", agent2_url=f"http://127.0.0.1:{p2}",
            tools="vision", use_llm_selector=True, vision_variant="fractions",
            events_dir=tmp_path / "runs", ledger_path=tmp_path / "l.jsonl"))
    finally:
        s1.should_exit = s2.should_exit = True
    ev = events_of(tmp_path, result)
    chosen = next(e for e in ev if e["type"] == "transport_selected")
    assert chosen["transport"] == "a2a" and chosen["agent1_url"] == f"http://127.0.0.1:{p1}"
    assert "transport_fallback" not in [e["type"] for e in ev]
    assert seen == [("vision", True, "fractions")]                       # the remote agent honoured the run's options


def test_dashboard_launcher_reads_agent_urls_and_auth_from_the_environment(demo, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from dashboard.server import create_app

    p1, p2 = free_port(), free_port()
    perception = InProcessPerception(PerceptionAgent(tools=[Fixed()], sleep=lambda s: None), max_workers=2)
    s1 = _serve(build_perception_app(perception, "127.0.0.1", p1), p1)
    s2 = _serve(build_planner_app(InProcessPlanner(), "127.0.0.1", p2), p2)
    monkeypatch.setenv("TRIAGE_AGENT1_URL", f"http://127.0.0.1:{p1}")
    monkeypatch.setenv("TRIAGE_AGENT2_URL", f"http://127.0.0.1:{p2}")
    monkeypatch.delenv("TRIAGE_A2A_AUTH", raising=False)
    try:
        c = TestClient(create_app(tmp_path / "runs", demo))
        assert c.get("/api/health").json()["agents"] == "remote"
        run_id = c.post("/api/runs", json={"disaster": "storm", "transport": "a2a", "wave_size": None}).json()["run_id"]
        deadline = time.time() + 30
        while time.time() < deadline and not c.get(f"/api/runs/{run_id}/events").json()["finished"]:
            time.sleep(0.2)
        ev = c.get(f"/api/runs/{run_id}/events").json()["events"]
        chosen = next(e for e in ev if e["type"] == "transport_selected")
        assert chosen["agent1_url"] == f"http://127.0.0.1:{p1}" and ev[-1]["type"] == "run_finished"
    finally:
        s1.should_exit = s2.should_exit = True


def test_without_agent_urls_the_dashboard_reports_local_agents(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from dashboard.server import create_app

    monkeypatch.delenv("TRIAGE_AGENT1_URL", raising=False)
    assert TestClient(create_app(tmp_path / "r", tmp_path / "d")).get("/api/health").json()["agents"] == "local"


def test_a_demo_folder_that_disagrees_with_demo_json_raises_a_warning_event(demo, tmp_path):
    (demo / "storm" / "images" / "storm_00000099_post_disaster.png").write_bytes(make_png())     # a stray extra tile
    result = execute_run(RunOptions(disaster="storm", demo_root=demo, offline=True, rounds=0,
                                    events_dir=tmp_path / "runs", ledger_path=tmp_path / "l.jsonl"))
    warning = next(e for e in events_of(tmp_path, result) if e["type"] == "warning")
    assert "6 tiles" in warning["message"] and "demo.json says 5" in warning["message"]
    assert "select_demo_tiles" in warning["message"]


def test_a_matching_demo_folder_raises_no_warning(demo, tmp_path):
    result = execute_run(RunOptions(disaster="storm", demo_root=demo, offline=True, rounds=0,
                                    events_dir=tmp_path / "runs", ledger_path=tmp_path / "l.jsonl"))
    assert "warning" not in [e["type"] for e in events_of(tmp_path, result)]


def test_remote_agents_do_not_construct_local_agents_unless_the_fallback_is_needed(demo, tmp_path, monkeypatch):
    import agents.orchestrator.runner as runner

    built = []
    real = runner.build_in_process_perception
    monkeypatch.setattr(runner, "build_in_process_perception", lambda *a, **k: built.append(a) or real(*a, **k))

    port1, port2 = free_port(), free_port()
    perception = InProcessPerception(PerceptionAgent(tools=[Fixed()], sleep=lambda s: None), max_workers=2)
    s1 = _serve(build_perception_app(perception, "127.0.0.1", port1), port1)
    s2 = _serve(build_planner_app(InProcessPlanner(), "127.0.0.1", port2), port2)
    try:
        execute_run(RunOptions(disaster="storm", demo_root=demo, offline=True, transport="a2a", rounds=0,
                               agent1_url=f"http://127.0.0.1:{port1}", agent2_url=f"http://127.0.0.1:{port2}",
                               events_dir=tmp_path / "runs", ledger_path=tmp_path / "l.jsonl"))
        assert built == []                                                    # healthy remote agents: nothing built locally
    finally:
        s1.should_exit = s2.should_exit = True

    dead = f"http://127.0.0.1:{free_port()}"
    result = execute_run(RunOptions(disaster="storm", demo_root=demo, offline=True, transport="a2a", rounds=0,
                                    agent1_url=dead, agent2_url=dead,
                                    events_dir=tmp_path / "runs2", ledger_path=tmp_path / "l2.jsonl"))
    assert len(built) == 1                                                    # built once, on the first failed call
    assert result.versions                                                    # and the run still completed
