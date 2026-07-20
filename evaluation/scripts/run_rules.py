"""Stage 3 · Leg 1 — rule runner + dev validation (RUN2_INSTRUCTIONS.md §3.1 steps 4-6).

Two entry points:
  validate  — run draft rules over dev, keep only precision≥0.5 AND ≥2 dev-positive hits
              (rest → rules/rejected.jsonl with reason). Writes rules/ruleset.v1/kept.jsonl
              + reports/leg1_rules.md. Success gate: ≥15 kept AND dev FPR ≤5%.
  detect    — expose leg1 as a harness detect(view, tools) fn (LLM-free), for fusion/ablation.

A rule fires on a VIEW's diff (added lines only — we match the code the commit introduces).
Rules match the antipattern form; we scan added ('+') lines of the PR-time diff.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mpb_common import BASE, CASES, PROMPTS, REPORTS, SPLITS, read_jsonl, now_iso  # noqa: E402
import eval_harness as H  # noqa: E402

RULES_DIR = BASE / "rules" / "ruleset.v1"


def _compile(rules):
    out = []
    for r in rules:
        if r.get("declined") or r.get("error"):
            continue
        if r.get("matcher_kind") != "regex" or not r.get("regex"):
            # ast rules: we keep the record but cannot run them dependency-free here;
            # mark as non-executable so validation drops them honestly (negative result).
            out.append((r, None))
            continue
        try:
            out.append((r, re.compile(r["regex"])))
        except re.error:
            out.append((r, None))
    return out


def _added_lines(diff_text):
    return "\n".join(l[1:] for l in diff_text.splitlines()
                     if l.startswith("+") and not l.startswith("+++"))


def _fire(compiled, view):
    """Return list of (rule, matched_text) that fire on this view's added lines."""
    added = _added_lines(view["diff"])
    files = view.get("changed_files", [])
    hits = []
    for r, rx in compiled:
        if rx is None:
            continue
        globs = r.get("target_globs") or []
        if globs and not any(any(Path(f).match(g) for g in globs) for f in files):
            continue
        m = rx.search(added)
        if m:
            hits.append((r, m.group(0)[:120]))
    return hits


def make_detect(ruleset_file=None):
    rules = list(read_jsonl(ruleset_file or (RULES_DIR / "kept.jsonl")))
    compiled = _compile(rules)

    def detect(view, tools):
        hits = _fire(compiled, view)
        findings = [{
            "severity": r.get("severity", "important"),
            "category": f"static-rule:{r.get('rule_id','?')}",
            "file": None, "line": None,
            "claim": f"static rule '{r.get('rule_id')}' matched antipattern: {r.get('antipattern','')} "
                     f"(matched: {mt!r})",
            "confidence": 0.7,
            "suggested_benchmark": None,
        } for r, mt in hits]
        return findings, {"n_turns": 0, "tokens": {"in": 0, "out": 0}, "cost": 0.0}
    return detect


def cmd_validate(a):
    draft = list(read_jsonl(RULES_DIR / (a.draft or "draft_rules.jsonl")))
    compiled = _compile(draft)
    executable = [(r, rx) for r, rx in compiled if rx is not None]
    print(f"draft rules: {len(draft)} ({sum(1 for r in draft if r.get('declined'))} declined), "
          f"{len(executable)} executable regex rules", flush=True)

    # use harness loaders for correctness (neg shas live in negatives_v011)
    H_cards, H_pairs, H_negs = H.load_cards(), H.load_pairs(), H.load_negs()
    loaded = H.load_items(SPLITS / "dev.txt", H_cards, H_pairs, H_negs)
    # restrict to a manageable but representative dev validation set for rule stats:
    # ALL perf positives + ALL negatives (rules are cheap/LLM-free, so run everything)
    pos = [it for it in loaded if it["kind"] == "case"
           and H_cards.get(it["item_id"].split(":", 1)[1], {}).get("is_perf_related")] \
        + [it for it in loaded if it["kind"] == "pair"]
    neg = [it for it in loaded if it["kind"] == "neg"]
    print(f"dev validation set: {len(pos)} positives, {len(neg)} negatives (LLM-free)", flush=True)

    # per-rule tallies + per-negative-item the set of rule_ids that fired (for FPR of kept)
    pos_hit = defaultdict(int)
    neg_hit = defaultdict(int)
    neg_fired_rules = []  # one set per negative item
    rid = lambda r: r.get("rule_id", "?")

    def scan(items, tally, record_sets=None):
        for i, it in enumerate(items):
            try:
                view = H.build_view(it["repo"], it["target_sha"])
            except Exception:
                if record_sets is not None:
                    record_sets.append(set())
                continue
            fired = {rid(r) for r, mt in _fire(executable, view)}
            for rname in fired:
                tally[rname] += 1
            if record_sets is not None:
                record_sets.append(fired)
            if (i + 1) % 500 == 0:
                print(f"    scanned {i+1}/{len(items)}", flush=True)
    print("  scanning positives...", flush=True)
    scan(pos, pos_hit)
    print("  scanning negatives...", flush=True)
    scan(neg, neg_hit, record_sets=neg_fired_rules)

    kept, rejected = [], []
    for r, rx in executable:
        i = rid(r)
        ph, nh = pos_hit.get(i, 0), neg_hit.get(i, 0)
        prec = ph / (ph + nh) if (ph + nh) else 0.0
        r = {**r, "dev_pos_hits": ph, "dev_neg_hits": nh, "dev_precision": round(prec, 3)}
        if prec >= 0.5 and ph >= 2:
            kept.append(r)
        else:
            r["reject_reason"] = (f"precision {prec:.2f}<0.5" if prec < 0.5 else "") + \
                                 (f" pos_hits {ph}<2" if ph < 2 else "")
            rejected.append(r)
    # ast/non-executable rules are rejected as non-runnable (honest negative result)
    for r, rx in compiled:
        if rx is None and not r.get("declined") and not r.get("error"):
            rejected.append({**r, "reject_reason": "non-executable (ast/invalid regex) — dropped"})

    out_kept = a.out or "kept.jsonl"
    (RULES_DIR / out_kept).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept))
    (BASE / "rules" / ("rejected_noMega.jsonl" if a.out else "rejected.jsonl")).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rejected))

    # FPR of kept ruleset = fraction of negatives where ANY kept rule fired (reuse the scan)
    kept_ids = {r.get("rule_id") for r in kept}
    tot_neg_fp = sum(1 for fired in neg_fired_rules if fired & kept_ids)
    dev_fpr = tot_neg_fp / len(neg) if neg else 0
    ok = len(kept) >= 15 and dev_fpr <= 0.05
    L = [f"# Leg 1 — static rules (metrics.v1 §3.1)", "",
         f"- generated: {now_iso()}",
         f"- mined {len(draft)} drafts → {len(executable)} executable → **{len(kept)} kept**, "
         f"{len(rejected)} rejected",
         f"- keep rule: dev precision ≥0.5 AND ≥2 dev-positive hits",
         f"- **dev FPR of kept ruleset: {dev_fpr:.1%}** (target ≤5%)",
         f"- success gate (≥15 kept AND FPR≤5%): {'✓ MET' if ok else '✗ NOT MET — recorded honestly, not relaxed'}",
         "", "## kept rules", "", "| rule_id | leaf | prec | pos | neg | severity |", "|---|---|--:|--:|--:|---|"]
    for r in sorted(kept, key=lambda x: -x["dev_precision"]):
        L.append(f"| {r.get('rule_id')} | {r.get('taxonomy_leaf','?')} | {r['dev_precision']} | "
                 f"{r['dev_pos_hits']} | {r['dev_neg_hits']} | {r.get('severity','important')} |")
    L += ["", f"rejected rules + reasons → `rules/rejected.jsonl` (负结果也是论文材料)"]
    rpt = "leg1_rules_noMega.md" if a.out else "leg1_rules.md"
    (REPORTS / rpt).write_text("\n".join(L) + "\n")
    print("\n".join(L[:9]))
    print(f"\nwrote rules/ruleset.v1/{out_kept} ({len(kept)}), reports/{rpt}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("--draft", default="draft_rules.jsonl")
    v.add_argument("--out", default=None, help="output kept filename (e.g. kept_noMega.jsonl)")
    a = ap.parse_args()
    if a.cmd == "validate":
        cmd_validate(a)


if __name__ == "__main__":
    main()
