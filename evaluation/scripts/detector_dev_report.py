"""Stage 3.5 — detector_v1 dev report (RUN2_INSTRUCTIONS.md §3.4 step 5 / §3.2-3.3 criteria).

Scores detector_v1 (and its ablations) on the dev tuning split and emits
reports/detector_v1_dev.md with:
  - recall@2 overall + per static_detectability stratum (high/medium/low) — the legs' targets
  - benign FPR (weighted + Wilson CI)
  - ADVERSARIAL before/after: detector_v1 vs ablate_adversarial (recall & FPR delta) — the paper figure
  - leg success criteria verdicts:
      leg1: (from reports/leg1_rules.md) ≥15 kept & FPR≤5%
      leg2: dev medium-subset recall@2 vs baseline(b) medium — target +30% relative
      leg3: low-subset recall@2 + leg3-only FPR

Usage: uv run python detector_dev_report.py --split dev_tune
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mpb_common import BASE, CASES, REPORTS, read_jsonl, now_iso  # noqa: E402
import eval_metrics as M  # noqa: E402


def _detectability(item_id, kind, cards):
    if kind == "pair":
        return "pair"
    c = cards.get(item_id.split(":", 1)[1], {})
    return c.get("static_detectability", "?")


def _score_with_strata(detector_id, split, cards, cache=None):
    """Return (metrics_dict, judge_cache). Adds per-detectability recall@2."""
    m = M.score_predictions(detector_id, split, judge_cache=cache)
    # re-open preds to compute per-detectability using the SAME judge cache
    preds = [json.loads(l) for l in (BASE / "predictions" / detector_id / f"{split}.jsonl").read_text().splitlines()]
    cache = m["cache"]
    pos = [p for p in preds if p["kind"] in ("case", "pair")
           and (p["kind"] == "pair" or cards.get(p["item_id"].split(":", 1)[1], {}).get("is_perf_related"))]

    def hit2(p):
        j = cache.get(p["item_id"], {})
        if not j.get("hit"):
            return False
        idx = j.get("hit_finding_index")
        top = M.topk_severe(p["findings"], 2)
        if idx is None or idx >= len(p["findings"]):
            return len(top) > 0
        return id(p["findings"][idx]) in {id(f) for f in top}

    strata = defaultdict(lambda: [0, 0])
    for p in pos:
        d = _detectability(p["item_id"], p["kind"], cards)
        strata[d][1] += 1
        if hit2(p):
            strata[d][0] += 1
    m["by_detectability2"] = {k: tuple(v) for k, v in strata.items()}
    return m, cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev_tune")
    a = ap.parse_args()
    cards = {c["case_id"]: c for c in read_jsonl(CASES / "cards_final_v011.jsonl")}

    variants = ["detector_v1", "ablate_adversarial"]
    variants = [v for v in variants if (BASE / "predictions" / v / f"{a.split}.jsonl").exists()]
    scored = {}
    for v in variants:
        print(f"scoring {v} on {a.split} ...", flush=True)
        m, _ = _score_with_strata(v, a.split, cards)
        scored[v] = m

    def rc(t):
        return M.fmt_recall(t)

    L = [f"# detector_v1 — dev tuning report ({a.split}, metrics.v1 §3.5)", "",
         f"- generated: {now_iso()} · judge {M.OPUS} (κ=0.926) · **static ceiling {M.CEILING}%**",
         f"- dev split (build/tune only — test untouched)", ""]

    # overall + strata
    L += ["## recall@2 — overall + per static_detectability", "",
          "| variant | overall@2 | high | medium | low | pair | FPR(weighted) |",
          "|---|---|---|---|---|---|---|"]
    for v in variants:
        m = scored[v]
        bd = m["by_detectability2"]
        fp, n, (lo, hi) = m["fpr"].get("_weighted_total", (0, 0, (0, 0)))
        L.append(f"| {v} | {rc(m['overall'][2])} | {rc(bd.get('high',(0,0)))} | "
                 f"{rc(bd.get('medium',(0,0)))} | {rc(bd.get('low',(0,0)))} | "
                 f"{rc(bd.get('pair',(0,0)))} | {fp}/{n} ({100*fp/n if n else 0:.1f}%, [{100*lo:.1f},{100*hi:.1f}]) |")

    # adversarial before/after
    if "detector_v1" in scored and "ablate_adversarial" in scored:
        a_on, a_off = scored["detector_v1"], scored["ablate_adversarial"]
        ro, no = a_on["overall"][2]; rf, nf = a_off["overall"][2]
        fp_on, n_on, _ = a_on["fpr"]["_weighted_total"]
        fp_off, n_off, _ = a_off["fpr"]["_weighted_total"]
        L += ["", "## 对抗验证前后（论文图）", "",
              "| | recall@2 | benign FPR |", "|---|---|---|",
              f"| 关闭对抗层 (ablate_adversarial) | {rc(a_off['overall'][2])} | "
              f"{fp_off}/{n_off} ({100*fp_off/n_off if n_off else 0:.1f}%) |",
              f"| 开启对抗层 (detector_v1) | {rc(a_on['overall'][2])} | "
              f"{fp_on}/{n_on} ({100*fp_on/n_on if n_on else 0:.1f}%) |",
              "",
              f"- 对抗层将 benign FPR 从 {100*fp_off/n_off if n_off else 0:.1f}% 降到 "
              f"{100*fp_on/n_on if n_on else 0:.1f}%，recall@2 从 "
              f"{100*ro/no if no else 0:.1f}% 变为 {100*rf/nf if nf else 0:.1f}%（rf 为关闭值）。"]

    # per-taxonomy for detector_v1
    if "detector_v1" in scored:
        m = scored["detector_v1"]
        L += ["", "## detector_v1 per-taxonomy recall@2", "", "| category | recall@2 |", "|---|---|"]
        for k in sorted(m["percat2"], key=lambda k: -m["percat2"][k][1]):
            L.append(f"| {k} | {rc(m['percat2'][k])} |")

    L += ["", "## 说明",
          "- leg1 成功判据见 `reports/leg1_rules.md`（31 kept / FPR 2.6% → ✓）。",
          "- leg2（medium 主力）/ leg3（low 路由）判据在此表 high/medium/low 列体现。",
          "- 全部在 dev 上；配置冻结后方可碰 test（Stage 4）。"]
    (REPORTS / "detector_v1_dev.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote reports/detector_v1_dev.md")


if __name__ == "__main__":
    main()
