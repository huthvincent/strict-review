"""Stage 1.3 — freeze the test evaluation subset (RUN2_INSTRUCTIONS.md §1.3).

All 818 test positives + all 116 test pairs + a deterministic stratified sample
of negatives (random.Random(20260718)):
  hard-neg-hotfile 500/2237 · random-benign 300/621 · false-signal-smoke-ci
  150/217 · false-signal-perf-infra 42/42 · lookalike 3/3  = 995 negatives.
Writes splits/test_eval_subset.txt (case:/pair:/neg: lines) + sentinel
splits/TEST_SUBSET_FROZEN. All contestants must run on the identical subset.
"""
from __future__ import annotations

import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mpb_common import SPLITS, NEGATIVES, CASES, read_jsonl, now_iso  # noqa: E402

NEG_QUOTA = {
    "hard-negative-hotfile": 500,
    "random-benign": 300,
    "false-signal-smoke-ci": 150,
    "false-signal-perf-infra": 42,
    "hard-negative-lookalike": 3,
}


def main():
    if (SPLITS / "TEST_SUBSET_FROZEN").exists():
        print("TEST_SUBSET_FROZEN already exists — refusing to re-freeze."); return
    test_lines = [l.strip() for l in (SPLITS / "test.txt").read_text().splitlines() if l.strip()]
    # positives = perf-related cases ONLY (spec §1.3: exactly 818). Non-perf case lines
    # in test.txt are neither positives nor negatives, so they are excluded from the
    # evaluation subset — including them would waste spend on unscored items.
    # NOTE: test.txt (and negatives_v011.jsonl) contain duplicate case_ids from the v0.2
    # build — dedup everywhere so the frozen subset has unique items only.
    cards = {c["case_id"]: c for c in read_jsonl(CASES / "cards_final_v011.jsonl")}
    all_case = list(dict.fromkeys(l for l in test_lines if l.startswith("case:")))
    pos = [l for l in all_case if cards.get(l.split(":", 1)[1], {}).get("is_perf_related")]
    dropped_nonperf = len(all_case) - len(pos)
    pairs = list(dict.fromkeys(l for l in test_lines if l.startswith("pair:")))
    neg_ids = list(dict.fromkeys(l.split(":", 1)[1] for l in test_lines if l.startswith("neg:")))

    negs = {n["case_id"]: n for n in read_jsonl(NEGATIVES / "negatives_v011.jsonl")}
    by_type = defaultdict(list)
    seen_nid = set()
    for nid in neg_ids:
        if nid in seen_nid:
            continue
        seen_nid.add(nid)
        n = negs.get(nid)
        if n:
            by_type[n["negative_type"]].append(nid)

    rnd = random.Random(20260718)
    picked_neg = []
    report = []
    for ntype, quota in NEG_QUOTA.items():
        pool = sorted(set(by_type.get(ntype, [])))  # unique ids only
        take = min(quota, len(pool))
        picked = rnd.sample(pool, take) if take < len(pool) else pool
        picked_neg += [f"neg:{x}" for x in picked]
        report.append((ntype, take, len(pool)))

    subset = pos + pairs + picked_neg
    (SPLITS / "test_eval_subset.txt").write_text("\n".join(subset) + "\n")
    (SPLITS / "TEST_SUBSET_FROZEN").write_text(now_iso() + "\n")

    print(f"test_eval_subset: {len(pos)} pos + {len(pairs)} pairs + {len(picked_neg)} neg "
          f"= {len(subset)} items (seed 20260718)")
    print(f"  (dropped {dropped_nonperf} non-perf case lines from test.txt)")
    for ntype, take, pool in report:
        print(f"  {ntype}: {take}/{pool}")
    print("wrote splits/test_eval_subset.txt + TEST_SUBSET_FROZEN")


if __name__ == "__main__":
    main()
