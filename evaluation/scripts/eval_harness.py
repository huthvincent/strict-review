"""Stage 1.1 — evaluation harness (RUN2_INSTRUCTIONS.md §1.1, harness.v1).

Turns a split item into a LEAK-FREE PR-time view and runs any detector against it.

Presentation rule (protocol.v1 §2, HARD): the detector sees ONLY the inducing
commit's point-in-time (diff, message, parent-tree snapshot). No posterior card
field (mechanism/evidence/magnitude/issue_refs/pr_ref/taxonomy_label/fix) ever
enters the view. The harness builds the view from git; it never hands a card to
a detector.

- case item → target sha = the case's sha
- pair item → target sha = **inducing_sha**
- view = {repo, sha, parent_sha, author_date, commit_message, diff, changed_files}
  diff = `git show <sha>` (first-parent for merges), trimmed via packer.v1
  (non-test hunks first, ≤8k tokens, records diff_truncated)
- detector tools, all anchored to the PARENT snapshot:
  read_file_at_parent(path) · grep_at_parent(pattern, glob) · git_log_before(path, n)
  any ref resolving to a commit at/after the target sha is REFUSED + leak_attempt=True

detect(view, tools) -> list[Finding]
  Finding = {severity, category, file, line, claim, confidence, suggested_benchmark}
→ predictions/{detector_id}/{split}.jsonl:
  {item_id, findings, n_turns, tokens, cost, latency_s, leak_attempt}
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mpb_common import BASE, REPOS, CASES, PAIRS, SPLITS, read_jsonl, repo_dir, git  # noqa: E402
from diff_packer import pack_commit  # noqa: E402

HARNESS_VERSION = "harness.v1"
VIEW_KEYS = {"repo", "sha", "parent_sha", "author_date", "commit_message",
             "diff", "changed_files", "diff_truncated"}
PRED_DIR = BASE / "predictions"


# ---------------------------------------------------------------------------
# item → target sha
# ---------------------------------------------------------------------------
def load_items(split_file: Path, cards: dict, pairs: dict, negs: dict | None = None) -> list[dict]:
    """Return [{item_id, kind: case|pair|neg, repo, target_sha}] for a split.
    case → card sha; pair → inducing_sha; neg → negative record sha (negatives
    live in negatives_v011.jsonl, not cards_final)."""
    negs = negs if negs is not None else load_negs()
    items = []
    for ln in Path(split_file).read_text().splitlines():
        ln = ln.strip()
        if not ln or ":" not in ln:
            continue
        typ, _id = ln.split(":", 1)
        if typ == "case":
            c = cards.get(_id)
            if c:
                items.append({"item_id": ln, "kind": "case", "repo": c["repo"],
                              "target_sha": c["sha"]})
        elif typ == "neg":
            n = negs.get(_id) or cards.get(_id)
            if n:
                items.append({"item_id": ln, "kind": "neg", "repo": n["repo"],
                              "target_sha": n["sha"]})
        elif typ == "pair":
            p = pairs.get(_id)
            if p:  # pair → inducing commit is the point-in-time
                items.append({"item_id": ln, "kind": "pair", "repo": p["repo"],
                              "target_sha": p["inducing_sha"]})
    return items


# ---------------------------------------------------------------------------
# leak-free PR-time view
# ---------------------------------------------------------------------------
def build_view(repo: str, sha: str) -> dict:
    parent = _first_parent(repo, sha)
    msg = git(repo, "show", "-s", "--format=%B", sha, timeout=60).strip()
    date = git(repo, "show", "-s", "--format=%aI", sha, timeout=60).strip()
    files = [f for f in git(repo, "show", "--first-parent", "--name-only",
                            "--format=", sha, timeout=90).splitlines() if f.strip()]
    packed = pack_commit(repo, sha, msg, files)
    # packed["text"] embeds message+stat+diff; for the view we want the raw diff only,
    # so rebuild a diff-only trimmed blob via the packer's budget on the diff hunks.
    diff = _trimmed_diff(repo, sha, files)
    return {
        "repo": repo, "sha": sha, "parent_sha": parent,
        "author_date": date, "commit_message": msg,
        "diff": diff["text"], "changed_files": files,
        "diff_truncated": diff["truncated"],
    }


def _first_parent(repo: str, sha: str) -> str:
    out = git(repo, "rev-list", "--parents", "-n", "1", sha, timeout=60).split()
    return out[1] if len(out) > 1 else ""  # parents follow the sha; [1] = first parent


DIFF_BUDGET_CHARS = int(8000 * 3.5)
TEST_RE = re.compile(r'((^|/)tests?/|_test\.py$|test_.*\.py$|functional_tests/)', re.I)


def _trimmed_diff(repo: str, sha: str, files: list[str]) -> dict:
    nontest = [f for f in files if not TEST_RE.search(f)]
    test = [f for f in files if TEST_RE.search(f)]
    pieces, used, truncated = [], 0, False
    for f in nontest + test:
        if used >= DIFF_BUDGET_CHARS:
            truncated = True
            break
        try:
            d = git(repo, "show", "--first-parent", "--no-color", "--format=", sha,
                    "--", f, timeout=90)
        except Exception:
            continue
        if not d.strip():
            continue
        if used + len(d) > DIFF_BUDGET_CHARS:
            d = d[:DIFF_BUDGET_CHARS - used] + "\n... [hunk truncated] ...\n"
            truncated = True
        pieces.append(d); used += len(d)
    return {"text": "\n".join(pieces), "truncated": truncated}


# ---------------------------------------------------------------------------
# parent-anchored tools (refuse any ref at/after the target sha)
# ---------------------------------------------------------------------------
class Tools:
    def __init__(self, repo: str, target_sha: str, parent_sha: str):
        self.repo = repo
        self.target = target_sha
        self.parent = parent_sha
        self.leak_attempt = False

    def _guard_commitish(self, ref: str) -> bool:
        """True if allowed. Anything that is the target or a descendant of it is a leak."""
        if not ref:
            return True
        try:
            full = git(self.repo, "rev-parse", ref, timeout=30).strip()
        except Exception:
            return True  # unresolvable ref → let the git call fail normally
        if full[:12] == self.target[:12]:
            self.leak_attempt = True
            return False
        # is `full` an ancestor of parent? allowed. else (== target or after) → refuse
        try:
            r = subprocess.run(["git", "-C", str(repo_dir(self.repo)), "merge-base",
                                "--is-ancestor", full, self.parent],
                               capture_output=True, timeout=30)
            if r.returncode != 0:  # full is NOT an ancestor of parent → at/after target
                self.leak_attempt = True
                return False
        except Exception:
            pass
        return True

    def read_file_at_parent(self, path: str) -> str:
        try:
            return git(self.repo, "show", f"{self.parent}:{path}", timeout=60)[:20000]
        except Exception as e:
            return f"[read error: {str(e)[:150]}]"

    def grep_at_parent(self, pattern: str, glob: str = "") -> str:
        args = ["grep", "-n", "-I", "-e", pattern, self.parent]
        if glob:
            args += ["--", glob]
        try:
            return git(self.repo, *args, timeout=60)[:15000]
        except Exception as e:
            return f"[grep: no matches or error: {str(e)[:100]}]"

    def git_log_before(self, path: str, n: int = 10) -> str:
        # log strictly before the target commit (parent and older)
        try:
            return git(self.repo, "log", "--format=%h %aI %s", f"-{int(n)}",
                       self.parent, "--", path, timeout=60)[:8000]
        except Exception as e:
            return f"[log error: {str(e)[:100]}]"


# ---------------------------------------------------------------------------
# run a detector over a split
# ---------------------------------------------------------------------------
def run_detector(detect_fn, detector_id: str, split: str, items: list[dict],
                 limit: int | None = None, on_result=None) -> Path:
    out = PRED_DIR / detector_id / f"{split}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["item_id"] for l in out.read_text().splitlines()} if out.exists() else set()
    todo = [it for it in items if it["item_id"] not in done]
    if limit:
        todo = todo[:limit]
    for it in todo:
        t0 = time.time()
        view = build_view(it["repo"], it["target_sha"])
        assert set(view) <= VIEW_KEYS, f"view leaked keys: {set(view) - VIEW_KEYS}"
        tools = Tools(it["repo"], it["target_sha"], view["parent_sha"])
        try:
            findings, meta = detect_fn(view, tools)
        except Exception as e:
            findings, meta = [], {"error": str(e)[:200]}
        rec = {"item_id": it["item_id"], "kind": it["kind"], "repo": it["repo"],
               "findings": findings, "leak_attempt": tools.leak_attempt,
               "latency_s": round(time.time() - t0, 2), **meta}
        with open(out, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if on_result:
            on_result(rec)
    return out


# convenience loaders
def load_cards():
    return {c["case_id"]: c for c in read_jsonl(CASES / "cards_final_v011.jsonl")}


def load_pairs():
    return {p["pair_id"]: p for p in read_jsonl(PAIRS / "regression_pairs_v011.jsonl")}


def load_negs():
    from mpb_common import NEGATIVES
    return {n["case_id"]: n for n in read_jsonl(NEGATIVES / "negatives_v011.jsonl")}


if __name__ == "__main__":
    # smoke: build one view and print its keys + a leak-guard demo
    cards, pairs = load_cards(), load_pairs()
    items = load_items(SPLITS / "dev.txt", cards, pairs)
    print(f"dev items: {len(items)} ({sum(1 for i in items if i['kind']=='case')} case, "
          f"{sum(1 for i in items if i['kind']=='pair')} pair, "
          f"{sum(1 for i in items if i['kind']=='neg')} neg)")
    it = next(i for i in items if i["kind"] == "case")
    v = build_view(it["repo"], it["target_sha"])
    print("view keys:", sorted(v))
    print("parent:", v["parent_sha"][:12], "diff chars:", len(v["diff"]),
          "trunc:", v["diff_truncated"])
