"""Deployment tooling: the staged build context, pinned requirements, and deploy.sh run against a
fake `gcloud` (so every flag that matters for security and correctness is checked without GCP)."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from importlib import metadata
from pathlib import Path

import pytest

from deploy.freeze_requirements import pins
from deploy.stage_context import stage

ROOT = Path(__file__).resolve().parents[2]
DEPLOY_SH = ROOT / "deploy" / "deploy.sh"
SMOKE_SH = ROOT / "deploy" / "smoke_test.sh"
needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")


# --- pinned requirements -----------------------------------------------------------------------------


def test_pins_use_installed_versions_and_the_adk_a2a_extra():
    versions = {"google-adk": "2.10.0", "a2a-sdk": "1.2.1", "pydantic": "2.13.5", "python-dotenv": "1.2.3"}

    def version(name):
        if name in versions:
            return versions[name]
        raise metadata.PackageNotFoundError(name)

    lines, missing, skipped = pins(version, required=["google-adk", "a2a-sdk", "pydantic", "networkx"],
                                   optional=["python-dotenv", "anthropic"])
    assert lines == ["google-adk[a2a]==2.10.0", "a2a-sdk==1.2.1", "pydantic==2.13.5", "python-dotenv==1.2.3"]
    assert missing == ["networkx"] and skipped == ["anthropic"]


# --- build context -----------------------------------------------------------------------------------


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    for d in ("agents/perception", "shared", "tools/vision", "scripts", "dashboard/static", "tests/unit", "deploy"):
        (r / d).mkdir(parents=True)
    (r / "agents/perception/agent.py").write_text("x")
    (r / "agents/perception/__pycache__").mkdir()
    (r / "agents/perception/__pycache__/agent.cpython-311.pyc").write_text("junk")
    (r / "shared/a.py").write_text("x"); (r / "tools/vision/client.py").write_text("x")
    (r / "scripts/serve_agents.py").write_text("x"); (r / "dashboard/static/app.js").write_text("x")
    (r / "tests/unit/test_x.py").write_text("tests are not shipped")
    (r / "deploy/Dockerfile").write_text("FROM python:3.11-slim"); (r / "deploy/requirements.txt").write_text("pydantic==2\n")
    (r / ".env").write_text("SECRET=1")
    for disaster in ("harvey", "wildfire"):
        d = r / "data" / "demo" / disaster
        for sub in ("images", "pre", "labels", ".thumbs"):
            (d / sub).mkdir(parents=True)
            (d / sub / "t.png").write_bytes(b"x" * 100)
        for f in ("geo.json", "demo.json", "footprints.json", "notes.txt"):
            (d / f).write_text("{}")
    (r / "data" / "cache" / "vision").mkdir(parents=True)
    (r / "data" / "cache" / "vision" / "classifier-abc.json").write_text("{}")
    (r / "data" / "runs" / "r1").mkdir(parents=True); (r / "data" / "runs" / "r1" / "events.jsonl").write_text("{}")
    (r / "data" / "xbd_real").mkdir(); (r / "data" / "xbd_real" / "huge.png").write_bytes(b"x" * 5_000_000)
    os.symlink(r / "data" / "xbd_real", r / "data" / "xbd_test")           # like the real 2.6 GB symlink
    return r


def files_under(p: Path) -> set[str]:
    return {str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()}


def test_stage_copies_only_what_the_services_need(repo, tmp_path):
    out = tmp_path / "ctx"
    info = stage(repo, out)
    got = files_under(out)
    assert {"Dockerfile", "requirements.txt", "agents/perception/agent.py", "shared/a.py", "tools/vision/client.py",
            "scripts/serve_agents.py", "dashboard/static/app.js", "data/cache/vision/classifier-abc.json"} <= got
    for d in ("harvey", "wildfire"):
        assert {f"data/demo/{d}/{x}" for x in ("images/t.png", "pre/t.png", "labels/t.png", "geo.json", "demo.json", "footprints.json")} <= got
    assert info["disasters"] == ["harvey", "wildfire"] and info["cache_entries"] == 1


def test_stage_never_includes_secrets_tests_runs_caches_or_the_xbd_data(repo, tmp_path):
    out = tmp_path / "ctx"
    stage(repo, out)
    got = files_under(out)
    joined = "\n".join(sorted(got))
    for banned in (".env", "tests/", "runs/", "xbd", "__pycache__", ".pyc", ".thumbs", "notes.txt"):
        assert banned not in joined, f"{banned!r} leaked into the build context"


def test_stage_can_limit_the_disasters_and_rejects_unknown_ones(repo, tmp_path):
    out = tmp_path / "ctx"
    stage(repo, out, ["harvey"])
    assert not (out / "data/demo/wildfire").exists() and (out / "data/demo/harvey/images/t.png").exists()
    with pytest.raises(ValueError, match="no demo data for: atlantis"):
        stage(repo, out, ["atlantis"])


def test_stage_clears_a_previous_context_and_checks_the_deploy_files_exist(repo, tmp_path):
    out = tmp_path / "ctx"
    out.mkdir(); (out / "stale.txt").write_text("old")
    stage(repo, out)
    assert not (out / "stale.txt").exists()
    (repo / "deploy/requirements.txt").unlink()
    with pytest.raises(FileNotFoundError, match="freeze_requirements"):
        stage(repo, out)


def test_stage_refuses_an_unexpectedly_large_context(repo, tmp_path):
    (repo / "data/demo/harvey/images/big.png").write_bytes(b"x" * 3_000_000)
    with pytest.raises(RuntimeError, match="limit 1 MB"):
        stage(repo, tmp_path / "ctx", max_mb=1)


# --- deploy.sh against a fake gcloud -----------------------------------------------------------------


FAKE_GCLOUD = r'''#!/usr/bin/env bash
echo "gcloud $*" >> "$FAKE_LOG"
case "$*" in
  "config get-value project") echo "${FAKE_PROJECT-proj-1}";;
  projects\ describe*) echo "123456789";;
  artifacts\ repositories\ describe*) [ "${FAKE_REPO_EXISTS-0}" = "1" ] || exit 1;;
  iam\ service-accounts\ describe*) [ "${FAKE_SAS_EXIST-0}" = "1" ] || exit 1;;
  run\ services\ describe*)
      svc="$4"; echo "${FAKE_ACTUAL_URL:-https://$svc-123456789.us-central1.run.app}";;
esac
exit 0
'''
FAKE_PYTHON = '#!/usr/bin/env bash\necho "python $*" >> "$FAKE_LOG"\n'


def run_deploy(tmp_path, **env):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for name, body in (("gcloud", FAKE_GCLOUD), ("python", FAKE_PYTHON)):
        f = bin_dir / name
        f.write_text(body); f.chmod(0o755)
    full = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "FAKE_LOG": str(tmp_path / "log"),
            "PROJECT": "proj-1", "REGION": "us-central1", **env}
    r = subprocess.run(["bash", str(DEPLOY_SH)], capture_output=True, text=True, env=full, timeout=60, cwd=ROOT)
    log = (tmp_path / "log").read_text().splitlines() if (tmp_path / "log").exists() else []
    return r, log


def index_of(log, needle):
    return next(i for i, line in enumerate(log) if needle in line)


@needs_bash
def test_scripts_have_valid_syntax():
    for script in (DEPLOY_SH, SMOKE_SH):
        assert subprocess.run(["bash", "-n", str(script)]).returncode == 0


@needs_bash
def test_full_deploy_runs_every_step_in_a_sensible_order(tmp_path):
    r, log = run_deploy(tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    order = ["services enable", "artifacts repositories create", "service-accounts create triage-agent1",
             "projects add-iam-policy-binding", "stage_context", "builds submit",
             "run deploy triage-agent1", "run deploy triage-agent2", "run deploy triage-dashboard",
             "run services add-iam-policy-binding triage-agent1"]
    positions = [index_of(log, n) for n in order]
    assert positions == sorted(positions), dict(zip(order, positions))
    assert "gcloud run services proxy triage-dashboard" in r.stdout


@needs_bash
def test_nothing_is_ever_made_public(tmp_path):
    r, log = run_deploy(tmp_path)
    deploys = [l for l in log if "run deploy" in l]
    assert len(deploys) == 3
    assert all("--no-allow-unauthenticated" in l for l in deploys)
    assert not any(re.search(r"(?<!no-)--allow-unauthenticated", l) for l in log)
    assert not any("allUsers" in l or "allAuthenticatedUsers" in l for l in log)


@needs_bash
def test_each_service_runs_as_its_own_service_account(tmp_path):
    _, log = run_deploy(tmp_path)
    sa = lambda svc: re.search(r"--service-account (\S+)", next(l for l in log if f"run deploy {svc}" in l)).group(1)
    accounts = {sa("triage-agent1"), sa("triage-agent2"), sa("triage-dashboard")}
    assert len(accounts) == 3
    assert sa("triage-agent1") == "triage-agent1@proj-1.iam.gserviceaccount.com"


@needs_bash
def test_roles_are_least_privilege(tmp_path):
    _, log = run_deploy(tmp_path)
    bindings = [l for l in log if "projects add-iam-policy-binding" in l]
    for_sa = lambda name: [re.search(r"--role (\S+)", l).group(1) for l in bindings if f"serviceAccount:{name}@" in l]
    assert for_sa("triage-agent1") == ["roles/aiplatform.user"]                    # Agent 1 may call Gemini, nothing else
    assert for_sa("triage-agent2") == []                                           # Agent 2 needs no Google API
    assert sorted(for_sa("triage-dashboard")) == ["roles/aiplatform.user", "roles/datastore.user"]
    assert not any("roles/owner" in l or "roles/editor" in l for l in bindings)


@needs_bash
def test_only_the_dashboard_may_invoke_the_agents(tmp_path):
    _, log = run_deploy(tmp_path)
    invokers = [l for l in log if "run services add-iam-policy-binding" in l]
    assert len(invokers) == 2
    for l in invokers:
        assert "--role roles/run.invoker" in l and "serviceAccount:triage-dashboard@proj-1.iam.gserviceaccount.com" in l
    assert {re.search(r"binding (\S+)", l).group(1) for l in invokers} == {"triage-agent1", "triage-agent2"}


@needs_bash
def test_dashboard_is_wired_to_the_agents_and_pinned_to_one_instance(tmp_path):
    _, log = run_deploy(tmp_path)
    dash = next(l for l in log if "run deploy triage-dashboard" in l)
    assert "--max-instances 1" in dash
    assert "TRIAGE_AGENT1_URL=https://triage-agent1-123456789.us-central1.run.app" in dash
    assert "TRIAGE_AGENT2_URL=https://triage-agent2-123456789.us-central1.run.app" in dash
    assert "TRIAGE_A2A_AUTH=id_token" in dash and "TRIAGE_STORE=firestore" in dash


@needs_bash
def test_each_agent_advertises_its_own_public_url_and_serves_the_right_agent(tmp_path):
    _, log = run_deploy(tmp_path)
    a1 = next(l for l in log if "run deploy triage-agent1" in l)
    a2 = next(l for l in log if "run deploy triage-agent2" in l)
    assert "TRIAGE_PUBLIC_URL=https://triage-agent1-123456789.us-central1.run.app" in a1 and "--agent,perception" in a1
    assert "TRIAGE_PUBLIC_URL=https://triage-agent2-123456789.us-central1.run.app" in a2 and "--agent,planner" in a2


@needs_bash
def test_dry_run_prints_the_plan_and_changes_nothing(tmp_path):
    r, log = run_deploy(tmp_path, DRY_RUN="1")
    assert r.returncode == 0 and "+ gcloud run deploy triage-dashboard" in r.stdout
    mutating = [l for l in log if any(x in l for x in ("run deploy", "services enable", "builds submit", "create ",
                                                       "add-iam-policy-binding", "stage_context"))]
    assert mutating == []


@needs_bash
def test_a_single_step_can_be_run_alone(tmp_path):
    r, log = run_deploy(tmp_path, STEP="build")
    assert [l for l in log if "builds submit" in l] and not [l for l in log if "run deploy" in l or "services enable" in l]


@needs_bash
def test_existing_resources_are_not_recreated(tmp_path):
    _, log = run_deploy(tmp_path, FAKE_REPO_EXISTS="1", FAKE_SAS_EXIST="1")
    assert not [l for l in log if "artifacts repositories create" in l or "service-accounts create" in l]


@needs_bash
def test_a_url_that_differs_from_the_predicted_one_is_flagged(tmp_path):
    r, _ = run_deploy(tmp_path, FAKE_ACTUAL_URL="https://something-else.run.app")
    assert "advertises the wrong address" in r.stdout and "TRIAGE_PUBLIC_URL=https://something-else.run.app" in r.stdout


@needs_bash
def test_missing_project_stops_immediately(tmp_path):
    r, log = run_deploy(tmp_path, PROJECT="", FAKE_PROJECT="")
    assert r.returncode == 1 and "No project" in r.stdout
    assert not [l for l in log if "run deploy" in l]
