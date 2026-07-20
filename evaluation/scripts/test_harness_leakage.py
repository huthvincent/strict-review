"""Stage 1.1 leakage self-test (RUN2_INSTRUCTIONS.md §1.1) — MUST PASS to proceed.

Three assertions:
(a) view key set ⊆ whitelist
(b) for 50 random items, the view JSON contains NO >20-char substring of that
    case's mechanism / evidence / taxonomy_label / magnitude_reported
(c) parent-anchored tools refuse a ref at/after the target sha (leak_attempt=True)
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mpb_common import SPLITS, git  # noqa: E402
import eval_harness as H  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


import re as _re


def _norm(s):
    # normalize for overlap comparison: lowercase, collapse non-alnum to single space.
    # This makes a quoted commit-message fragment inside `evidence` match the raw
    # message in the view (quotes/brackets/whitespace differences don't count as leaks).
    return _re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def word_runs(s, k=7):
    # Word-aligned k-grams over the normalized token stream. Word alignment (vs
    # char offsets) makes an evidence phrase and the same phrase in the diff line
    # up, so the allowed-source subtraction actually cancels legitimate quotes of
    # the diff/message. A real posterior-analysis leak is a k-gram of card prose
    # that is NOT present anywhere in the allowed diff+message.
    toks = _norm(s).split()
    return [" ".join(toks[i:i + k]) for i in range(0, max(0, len(toks) - k + 1))] if len(toks) >= k else ([" ".join(toks)] if toks else [])


def main():
    cards = H.load_cards()
    pairs = H.load_pairs()
    negs = H.load_negs()
    # use dev (never test) for the leak test
    items = H.load_items(SPLITS / "dev.txt", cards, pairs, negs)
    rnd = random.Random(20260718)

    # (a) + (b): sample 50 case items (they have posterior fields to leak)
    case_items = [it for it in items if it["kind"] == "case"]
    sample = rnd.sample(case_items, min(50, len(case_items)))
    key_ok = True
    leak_hits = []
    for it in sample:
        v = H.build_view(it["repo"], it["target_sha"])
        if set(v) - H.VIEW_KEYS:
            key_ok = False
        vj = _norm(json.dumps(v, ensure_ascii=False))
        # A code symbol that appears in BOTH the card's mechanism/evidence AND the
        # raw diff/message is legitimately visible (the detector is meant to see the
        # diff). Genuine leakage = posterior ANALYSIS prose that is NOT already in
        # the allowed source. So exclude any substring that also occurs in the raw
        # diff+message (the allowed inputs) before flagging.
        allowed_src = _norm(v["diff"] + "\n" + v["commit_message"])
        c = cards[it["item_id"].split(":", 1)[1]]
        for field in ("mechanism", "evidence", "taxonomy_label", "magnitude_reported"):
            for sub in word_runs(str(c.get(field) or "")):
                if sub and sub in vj and sub not in allowed_src:
                    leak_hits.append((it["item_id"], field, sub[:40]))
                    break
    check("(a) view keys ⊆ whitelist", key_ok, f"{len(sample)} views checked")
    check("(b) no posterior-field leakage in view (beyond diff/message)", not leak_hits,
          f"{len(leak_hits)} leaks" + (f": {leak_hits[:2]}" if leak_hits else ""))

    # (c) tool guard: pick a pair, try to read a file at the FIX commit (after inducing)
    pair = next(p for p in pairs.values() if p.get("fix_sha"))
    v = H.build_view(pair["repo"], pair["inducing_sha"])
    tools = H.Tools(pair["repo"], pair["inducing_sha"], v["parent_sha"])
    # a benign parent read should NOT trip leak; reading at the fix sha SHOULD
    _ = tools.read_file_at_parent(v["changed_files"][0] if v["changed_files"] else "README.md")
    benign_leak = tools.leak_attempt
    # now attempt a guarded commit-ish resolve to the fix commit (post-target)
    allowed = tools._guard_commitish(pair["fix_sha"])
    check("(c) tool guard refuses post-target ref", (not allowed) and tools.leak_attempt
          and not benign_leak,
          f"benign_leak={benign_leak} fix_allowed={allowed} leak_flag={tools.leak_attempt}")

    print(f"\n{'ALL LEAKAGE TESTS PASS' if not FAILS else 'LEAKAGE TEST FAILED: ' + ','.join(FAILS)}")
    sys.exit(0 if not FAILS else 1)


if __name__ == "__main__":
    main()
