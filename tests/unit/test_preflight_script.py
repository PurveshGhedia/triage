"""deploy/preflight.sh against a fake `gcloud` and `curl`, so each outcome can be exercised without GCP."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "deploy" / "preflight.sh"
pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")

FAKE_GCLOUD = r'''#!/usr/bin/env bash
echo "gcloud $*" >> "$FAKE_LOG"
case "$*" in
  "config get-value project") echo "${FAKE_PROJECT-proj-1}";;
  auth\ list*) echo "${FAKE_ACCOUNT-me@gmail.com}";;
  billing\ projects\ describe*) echo "${FAKE_BILLING-True}";;
  services\ list*)
      all="$*"; api="${all##*config.name=}"; api="${api%% *}"
      case " ${FAKE_APIS} " in *" $api "*) echo "$api";; esac;;
  resource-manager\ org-policies*) echo "${FAKE_ORGPOLICY-}";;
  run\ deploy*) [ "${FAKE_DEPLOY_OK-1}" = "1" ] || { echo "ERROR: deploy failed" >&2; exit 1; }; echo created > "$FAKE_STATE";;
  run\ services\ describe*) [ -f "$FAKE_STATE" ] && echo "https://hello.example.run.app" || exit 1;;
  run\ services\ delete*) rm -f "$FAKE_STATE"; echo deleted >> "$FAKE_LOG";;
  auth\ print-identity-token) echo "fake-token";;
esac
'''
FAKE_CURL = r'''#!/usr/bin/env bash
case "$*" in
  *Authorization*) printf '%s' "${FAKE_PRIVATE_CODE-200}";;
  *) printf '%s' "${FAKE_PUBLIC_CODE-200}";;
esac
'''
ALL_APIS = "run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com firestore.googleapis.com aiplatform.googleapis.com"


def run_script(tmp_path, **env):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for name, body in (("gcloud", FAKE_GCLOUD), ("curl", FAKE_CURL)):
        f = bin_dir / name
        f.write_text(body)
        f.chmod(0o755)
    full = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "FAKE_LOG": str(tmp_path / "log"),
            "FAKE_STATE": str(tmp_path / "state"), "FAKE_APIS": ALL_APIS, **env}
    r = subprocess.run(["bash", str(SCRIPT)], capture_output=True, text=True, env=full, timeout=60)
    log = (tmp_path / "log").read_text() if (tmp_path / "log").exists() else ""
    return r, log


def test_script_has_valid_bash_syntax():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0


def test_everything_fine_and_public_access_works(tmp_path):
    r, log = run_script(tmp_path)
    assert r.returncode == 0 and "FAILED: 0" in r.stdout
    assert "PUBLIC ACCESS works" in r.stdout
    assert "deleted" in log and not (tmp_path / "state").exists()            # throwaway service removed


def test_public_blocked_but_private_works_gives_the_proxy_instructions(tmp_path):
    r, log = run_script(tmp_path, FAKE_PUBLIC_CODE="403",
                        FAKE_ORGPOLICY="listPolicy:\n  allowedValues:\n  - C0123abcd")
    assert r.returncode == 1
    assert "domain-restricted-sharing policy is active" in r.stdout
    assert "PUBLIC ACCESS blocked, private works" in r.stdout
    assert "gcloud run services proxy triage-dashboard" in r.stdout
    assert "deleted" in log


def test_missing_apis_are_listed_with_the_enable_command_and_nothing_is_deployed(tmp_path):
    r, log = run_script(tmp_path, FAKE_APIS="firestore.googleapis.com")
    assert "run.googleapis.com NOT enabled" in r.stdout
    assert "gcloud services enable" in r.stdout and "run.googleapis.com" in r.stdout
    assert "Enable the missing APIs first" in r.stdout
    assert "run deploy" not in log                                           # never deploys without the Run API


def test_failed_deploy_is_reported_and_cleaned_up(tmp_path):
    (tmp_path / "state").write_text("exists")                                # the service exists despite the error
    r, log = run_script(tmp_path, FAKE_DEPLOY_OK="0", FAKE_PUBLIC_CODE="403", FAKE_PRIVATE_CODE="403")
    assert "deploy reported errors" in r.stdout
    assert "deleted" in log


def test_billing_off_and_no_account_are_flagged(tmp_path):
    r, _ = run_script(tmp_path, FAKE_BILLING="False", FAKE_ACCOUNT="")
    assert "billing does not look enabled" in r.stdout and "not signed in" in r.stdout
    assert r.returncode == 1


def test_no_project_stops_early_without_touching_anything(tmp_path):
    r, log = run_script(tmp_path, FAKE_PROJECT="")
    assert "no project set" in r.stdout and r.returncode == 1
    assert "run deploy" not in log


def test_no_org_policy_is_reported_as_ok(tmp_path):
    r, _ = run_script(tmp_path)
    assert "no domain-restriction policy found" in r.stdout
