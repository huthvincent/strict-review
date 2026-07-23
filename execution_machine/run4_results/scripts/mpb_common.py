"""MegaPerfBench Phase-0 shared utilities.

Single source of truth for paths, repo list, JSONL idempotent I/O, and the
Opus-4.8 model id. Every pipeline script imports from here so a machine move is
a one-line edit (CONSTITUTION §4: no hard-coded absolute paths in logic, all
LLM calls funnel through one gateway).

Storage discipline (intro.md §8.1): everything writes under BASE only; never
push to GitHub or any external service. Repo clones are *inputs*, kept on local
NVMe (REPOS), not under BASE.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(os.environ.get("MPB_BASE", "/mnt/efs/tsfm/rui/GAA/ai_infra"))
REPOS = Path(os.environ.get("MPB_REPOS", "/home/ec2-user/megaperf_repos"))

RAW = BASE / "raw"
SCREENING = BASE / "screening"
CASES = BASE / "cases"
PAIRS = BASE / "pairs"
NEGATIVES = BASE / "negatives"
REPORTS = BASE / "reports"
SCHEMAS = BASE / "schemas"
PROMPTS = BASE / "prompts"
SPLITS = BASE / "splits"
TAXONOMY = BASE / "taxonomy"
LOGS = BASE / "logs"

# repo display-name -> clone dir name (kept identical for simplicity)
REPOS_LIST = ["Megatron-LM", "vllm", "DeepSpeed", "TransformerEngine"]

# canonical short repo key used inside ids (§8.2: case_id = {repo}@{sha12})
REPO_KEY = {
    "Megatron-LM": "Megatron-LM",
    "vllm": "vllm",
    "DeepSpeed": "DeepSpeed",
    "TransformerEngine": "TransformerEngine",
}

# ---------------------------------------------------------------------------
# Model discipline (intro.md model rule): ALL LLM calls use Opus 4.8.
# On this box the Bedrock inference-profile id is us.anthropic.claude-opus-4-8.
# ---------------------------------------------------------------------------
OPUS_48 = os.environ.get("MPB_OPUS_MODEL", "us.anthropic.claude-opus-4-8")


def repo_dir(repo: str) -> Path:
    return REPOS / repo


# ---------------------------------------------------------------------------
# git helpers (offline; full clones so no network)
# ---------------------------------------------------------------------------
def git(repo: str, *args: str, timeout: int = 120) -> str:
    """Run a git command in a repo clone, offline. Returns stdout (str)."""
    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"  # never prompt / hit network for creds
    out = subprocess.run(
        ["git", "-C", str(repo_dir(repo)), *args],
        capture_output=True, text=True, timeout=timeout, env=env,
        errors="replace",
    )
    if out.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed in {repo}: {out.stderr[:500]}")
    return out.stdout


# ---------------------------------------------------------------------------
# JSONL idempotent I/O (intro.md §7.2: idempotent, resumable by key)
# ---------------------------------------------------------------------------
def read_jsonl(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_keys(path: Path, key: str) -> set[str]:
    """Return the set of `key` values already present in a JSONL file, so a
    rerun can skip completed records (idempotency)."""
    done = set()
    if not Path(path).exists():
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)[key])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def append_jsonl(path: Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    """Overwrite a JSONL file atomically (write temp then rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def now_iso() -> str:
    """ISO8601 UTC timestamp. Isolated here so scripts don't sprinkle
    datetime.now() calls (also keeps a single spot to stub in tests)."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
