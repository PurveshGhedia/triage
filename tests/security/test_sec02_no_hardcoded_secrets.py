"""SEC-02 (partial coverage): no API keys, service-account tokens, or private
keys should ever be hardcoded in the source tree.

IMPORTANT SCOPE NOTE: the test case as written in TRIAGE_Test_Cases.docx asks
to "inspect Cloud Logging output and agent responses after a normal run" --
that requires a live, deployed run, which doesn't exist until Terraform
provisioning happens. This test covers a different, real, and currently
testable part of the same intent: proving no secret is hardcoded anywhere in
the codebase itself, which for a solo project is the most likely place a
credential would actually leak. The Cloud Logging half of SEC-02 stays open
until infra is provisioned.

Why this test is meaningful given the project's actual design: nothing in
the codebase should EVER match these patterns, because Vertex AI / Firestore
/ BigQuery auth all go through Application Default Credentials (`gcloud auth
application-default login`), never a hardcoded API key -- this test is what
proves that design intent held in practice, not just in theory.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # tests/security/ -> repo root

SECRET_PATTERNS = {
    "Google API key": re.compile(r"AIza[0-9A-Za-z_\-]{35}"),
    "Anthropic API key": re.compile(r"sk-ant-[0-9A-Za-z_\-]{20,}"),
    "OpenAI-style API key": re.compile(r"sk-[0-9A-Za-z]{32,}"),
    "PEM private key header": re.compile(r"-----BEGIN (RSA |EC )?PRIVATE KEY-----"),
    "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
}

SCAN_EXTENSIONS = {".py", ".md", ".yml", ".yaml", ".json", ".txt"}
SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", "data"}


def iter_source_files():
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix not in SCAN_EXTENSIONS:
            continue
        if path.name == ".env":  # the one file allowed to hold real secrets -- gitignored
            continue
        yield path


def test_no_hardcoded_secrets_in_tracked_source():
    findings = []
    for path in iter_source_files():
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                findings.append(f"{label} pattern found in {path.relative_to(REPO_ROOT)}")

    assert not findings, "Possible hardcoded secret(s):\n" + "\n".join(findings)
