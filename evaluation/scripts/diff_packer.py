"""Deterministic, versioned diff packer (intro.md §3 Tier-1, §7.5).

Turns a commit into the text block fed to the Tier-1 classifier:
  - commit message (full) + author + date
  - `git show --stat` file list
  - diff, trimmed to a token budget, NON-TEST hunks first (§3 "优先保留非测试
    代码的 hunk"); records diff_truncated when anything is dropped.

Token budgeting: the plan (§7.7) forbids tiktoken and wants the count_tokens API
for *final* calibration. For the packer's trim decision we use a cheap, stable
char/CHARS_PER_TOK heuristic (trimming only needs to be roughly right and must
be deterministic + offline); the *actual* token usage is always read back exact
from the API response and reported. PACKER_VERSION bumps on any rule change.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mpb_common import git  # noqa: E402

PACKER_VERSION = "packer.v1"
DIFF_TOKEN_BUDGET = 6000          # §3: diff total ≤ 6k tokens
CHARS_PER_TOK = 3.5               # rough; only for trim decisions, never billed
TEST_RE = re.compile(r'((^|/)tests?/|_test\.py$|test_.*\.py$|(^|/)qa/|functional_tests/)', re.I)


def _budget_chars() -> int:
    return int(DIFF_TOKEN_BUDGET * CHARS_PER_TOK)


def pack_commit(repo: str, sha: str, message: str, files: list[str]) -> dict:
    """Return {text, diff_truncated, n_files, approx_diff_tokens}."""
    # per-file diffs; split test vs non-test, non-test first
    nontest = [f for f in files if not TEST_RE.search(f)]
    test = [f for f in files if TEST_RE.search(f)]
    ordered = nontest + test

    stat = git(repo, "show", "--stat", "--oneline", "--no-color",
               "-M", sha, "--", *files, timeout=120) if files else ""
    # keep only the --stat block (drop the diff that --stat's parent format adds)
    stat_lines = []
    for ln in stat.splitlines():
        stat_lines.append(ln)
        if re.search(r'\d+ files? changed', ln) or re.search(r'1 file changed', ln):
            break
    stat_block = "\n".join(stat_lines)

    budget = _budget_chars()
    pieces, used, truncated = [], 0, False
    for f in ordered:
        if used >= budget:
            truncated = True
            break
        try:
            d = git(repo, "show", "--no-color", "-M", "--format=", sha, "--", f,
                    timeout=120)
        except Exception:
            continue
        if not d.strip():
            continue
        remaining = budget - used
        if len(d) > remaining:
            d = d[:remaining] + "\n... [hunk truncated] ...\n"
            truncated = True
        pieces.append(d)
        used += len(d)
    if len(ordered) > 0 and used == 0:
        truncated = truncated  # nothing packed (e.g. binary-only)

    diff_text = "\n".join(pieces)
    text = (
        f"COMMIT {repo}@{sha[:12]}\n"
        f"=== MESSAGE ===\n{message.strip()}\n\n"
        f"=== FILES CHANGED (git show --stat) ===\n{stat_block.strip()}\n\n"
        f"=== DIFF (non-test hunks first; may be truncated) ===\n{diff_text.strip()}\n"
    )
    return {
        "text": text,
        "diff_truncated": truncated,
        "n_files": len(files),
        "approx_diff_tokens": int(used / CHARS_PER_TOK),
        "packer_version": PACKER_VERSION,
    }
