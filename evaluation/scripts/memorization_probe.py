"""Stage B — memorization probe (DATASET_COMPLETION_INSTRUCTIONS.md Stage B; protocol §3).

For every pair in the test split + a deterministic 10% stratified sample of
test-split perf-related cards: ask BARE Opus 4.8 (NO tools, NO diff, NO context —
only repo/sha/subject) whether it recalls the commit and its perf consequence.
Flag memorized=true only when the model self-reports recall confidence ≥0.5 AND
its answer is checkable against the record (names the real fix/culprit/symptom).

Grading uses the record we hold (fix_sha / symptom / culprit) as the answer key,
compared against the model's free recall — the model never sees the key.

Side annotation: writes splits/memorization_flags.jsonl (does NOT edit splits).

Usage: uv run python memorization_probe.py --workers 6 --max-cost-usd 30
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mpb_common import (  # noqa: E402
    SPLITS, CASES, PAIRS, RAW, REPOS_LIST, read_jsonl, load_keys, append_jsonl, now_iso,
)

OPUS = "us.anthropic.claude-opus-4-8"
PROMPT_VERSION = "memorization_probe.v1"
PRICE_IN, PRICE_OUT = 5.0, 25.0
OUT = SPLITS / "memorization_flags.jsonl"
PRN = re.compile(r"#(\d+)")

_write_lock = threading.Lock()
_cost_lock = threading.Lock()

EMIT = {
    "name": "emit_recall",
    "description": "Report whether you recall this specific commit and its perf consequence.",
    "input_schema": {"type": "object", "properties": {
        "recalls_commit": {"type": "boolean"},
        "perf_problem_known": {"type": ["string", "null"],
            "description": "if recalled: the specific symptom/mechanism, and the fix PR/commit if known; else null"},
        "recall_confidence": {"type": "number", "minimum": 0, "maximum": 1,
            "description": "confidence this comes from actual recall of THIS commit, not inference from the subject line"},
    }, "required": ["recalls_commit", "recall_confidence"]},
}

SYSTEM = ("You are being tested for prior knowledge of a specific public commit. You have NO "
          "tools and NO repository access — answer only from what you already know. If you do "
          "not recognize the commit, say so; do not guess or speculate from the subject line. "
          "Either you recall this specific commit or you do not. Call emit_recall once.")

_client = None


def _cl():
    global _client
    if _client is None:
        from anthropic import AnthropicBedrock
        _client = AnthropicBedrock(aws_region=os.environ.get("AWS_REGION", "us-east-1"))
    return _client


def probe(repo: str, sha: str, subject: str):
    user = (f"Commit {repo}@{sha} — subject: \"{subject}\".\n\n"
            f"1. Do you specifically recall this commit?\n"
            f"2. If yes: was it later found to cause a performance problem (throughput/latency/"
            f"memory/gpu-util/hang/compile-time)? What mechanism, and which later commit fixed it?\n"
            f"3. Your confidence this is actual recall of THIS commit (not subject-line inference).")
    resp = _cl().messages.create(model=OPUS, max_tokens=800, system=SYSTEM,
                                 messages=[{"role": "user", "content": user}],
                                 tools=[EMIT], tool_choice={"type": "tool", "name": "emit_recall"})
    inp = None
    for b in resp.content:
        if b.type == "tool_use":
            inp = b.input; break
    itok, otok = resp.usage.input_tokens, resp.usage.output_tokens
    return inp, (itok * PRICE_IN + otok * PRICE_OUT) / 1e6, itok, otok


def grade(inp: dict, answer_key: dict) -> tuple[bool, str]:
    """memorized iff recall_confidence>=0.5 AND the answer names a real, checkable fact."""
    if not inp or not inp.get("recalls_commit") or inp.get("recall_confidence", 0) < 0.5:
        return False, "no confident recall"
    ans = (inp.get("perf_problem_known") or "").lower()
    if not ans:
        return False, "recalled commit but no specific perf fact"
    # checkable signals: names the real fix PR#, or the real symptom
    hits = []
    for pr in answer_key.get("fix_prs", []):
        if f"#{pr}" in ans or f"pr {pr}" in ans:
            hits.append(f"fix#{pr}")
    for pr in answer_key.get("culprit_prs", []):
        if f"#{pr}" in ans:
            hits.append(f"culprit#{pr}")
    sym = answer_key.get("symptom")
    if sym and sym != "n/a" and sym.split("-")[0] in ans:
        hits.append(f"symptom:{sym}")
    if hits:
        return True, "verifiable: " + ",".join(hits)
    return False, "recall claimed but not verifiable against record"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-cost-usd", type=float, default=30.0)
    ap.add_argument("--full", action="store_true",
                    help="probe ALL test perf-related cards (spec §1.4), not just 10%%")
    a = ap.parse_args()

    test_ids = [l.strip() for l in open(SPLITS / "test.txt") if l.strip()]
    cards = {c["case_id"]: c for c in read_jsonl(CASES / "cards_final_v011.jsonl")}
    pairs = {p["pair_id"]: p for p in read_jsonl(PAIRS / "regression_pairs_v011.jsonl")}
    commits = {}
    for rp in REPOS_LIST:
        for c in read_jsonl(RAW / "commits" / f"{rp}.jsonl"):
            commits[c["id"]] = c

    # build targets: all test pairs + deterministic 10% of test perf-related cards
    targets = []  # (target_id, repo, sha, subject, answer_key)
    for tid in test_ids:
        kind, _id = tid.split(":", 1)
        if kind == "pair":
            p = pairs.get(_id)
            if not p:
                continue
            fixc = commits.get(f"{p['repo']}@{p['fix_sha'][:12]}", {})
            key = {"symptom": p.get("symptom"),
                   "fix_prs": PRN.findall(fixc.get("subject", "")),
                   "culprit_prs": []}
            targets.append((tid, p["repo"], p["fix_sha"], fixc.get("subject", ""), key))
    # 10% stratified (by repo) of test perf-related cards
    perf_cards = [tid.split(":", 1)[1] for tid in test_ids if tid.startswith("case:")
                  and cards.get(tid.split(":", 1)[1], {}).get("is_perf_related")]
    by_repo = {}
    for cid in perf_cards:
        by_repo.setdefault(cards[cid]["repo"], []).append(cid)
    for rp, ids in by_repo.items():
        ids = sorted(ids)
        if a.full:
            chosen = ids  # spec §1.4: probe every test perf-related card
        else:
            k = max(1, len(ids) // 10)
            step = max(1, len(ids) // k)
            chosen = ids[::step][:k]
        for cid in chosen:
            c = cards[cid]
            key = {"symptom": c.get("symptom"),
                   "fix_prs": PRN.findall(c.get("subject", "")) + PRN.findall(str(c.get("pr_ref") or "")),
                   "culprit_prs": PRN.findall(str(c.get("inducing_ref") or ""))}
            targets.append((f"case:{cid}", c["repo"], c["sha"], c.get("subject", ""), key))

    done = load_keys(OUT, "target_id")
    todo = [t for t in targets if t[0] not in done]
    print(f"memorization probe: {len(targets)} targets ({len(done)} done, {len(todo)} to run), "
          f"{a.workers} workers", flush=True)

    run_id = f"memprobe_{now_iso()}"
    state = {"spent": 0.0, "n": 0, "mem": 0, "err": 0, "stop": False}
    buf = []

    def work(t):
        if state["stop"]:
            return None
        tid, repo, sha, subj, key = t
        inp, cost, itok, otok = probe(repo, sha, subj)
        memd, why = grade(inp, key)
        return {"target_id": tid, "repo": repo, "sha": sha,
                "memorized": memd, "grade_reason": why,
                "model_recall": inp,
                "provenance": {"model": OPUS, "prompt_version": PROMPT_VERSION,
                               "run_id": run_id, "ts": now_iso(),
                               "tokens": {"in": itok, "out": otok}, "channel": "standard"}}, cost

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(work, t): t for t in todo}
        for fut in as_completed(futs):
            if state["stop"]:
                break
            try:
                res = fut.result()
            except Exception as e:
                state["err"] += 1
                if state["err"] <= 10:
                    print(f"  [err] {futs[fut][0]}: {str(e)[:90]}", flush=True)
                continue
            if res is None:
                continue
            rec, cost = res
            with _cost_lock:
                state["spent"] += cost; state["n"] += 1
                if rec["memorized"]:
                    state["mem"] += 1
                if state["spent"] >= a.max_cost_usd:
                    state["stop"] = True
            with _write_lock:
                buf.append(rec)
                if len(buf) >= 20:
                    append_jsonl(OUT, buf); buf = []
    with _write_lock:
        if buf:
            append_jsonl(OUT, buf)
    print(f"DONE: {state['n']} probed, {state['mem']} memorized, errors={state['err']}, "
          f"~${state['spent']:.2f}", flush=True)


if __name__ == "__main__":
    main()
