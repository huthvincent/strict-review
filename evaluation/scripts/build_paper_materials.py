"""Stage 8 — paper materials (RUN2_INSTRUCTIONS.md §8).

Assembles paper/ from the scored reports (judge cache makes re-scoring free):
  paper/tables/       *.csv + *.md  (baseline compare, ablation, per-kind, per-taxonomy,
                                     stratified FPR, cross-repo, dataset stats)
  paper/figures_data/ *.csv         (funnel counts, taxonomy dist, detectability dist,
                                     adversarial recall-FPR, recall@budget curve)
  paper/claims.md                   claims + supporting files/numbers + known rebuttals
  paper/limitations.md              limitations + mitigations

Pulls numbers from eval_metrics.score_predictions (cached) + existing reports.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mpb_common import BASE, CASES, read_jsonl, now_iso  # noqa: E402
import eval_metrics as M  # noqa: E402

PAPER = BASE / "paper"
TABLES = PAPER / "tables"
FIGS = PAPER / "figures_data"
FULL = "test_eval_subset"
ABL = "test_ablation_subset"


def _write_csv(path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def _md_table(header, rows):
    L = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        L.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(L)


def _rrate(t):
    h, n = t
    return round(100 * h / n, 1) if n else 0.0


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    cards = {c["case_id"]: c for c in read_jsonl(CASES / "cards_final_v011.jsonl")}

    # ---- score everything (cached → fast) ----
    print("scoring detectors (cached)...", flush=True)
    det = {}
    for d in ["detector_v1", "baseline_megatron", "baseline_generic", "baseline_keyword", "baseline_xfamily"]:
        if (BASE / "predictions" / d / f"{FULL}.jsonl").exists():
            det[d] = M.score_predictions(d, FULL)
    abl = {}
    for d in ["ablate_leg1", "ablate_leg2", "ablate_leg3", "ablate_adversarial"]:
        if (BASE / "predictions" / d / f"{ABL}.jsonl").exists():
            abl[d] = M.score_predictions(d, ABL)
    # detector_v1 restricted to ablation subset
    abl_ids = set(l.strip() for l in (BASE / "splits" / f"{ABL}.txt").read_text().splitlines() if l.strip())
    rows_r = [json.loads(l) for l in (BASE / "predictions" / "detector_v1" / f"{FULL}.jsonl").read_text().splitlines()
              if json.loads(l)["item_id"] in abl_ids]
    tmp = BASE / "predictions" / "detector_v1__paper"
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / f"{FULL}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows_r))
    abl["detector_v1"] = M.score_predictions("detector_v1__paper", FULL)

    NAMES = {"detector_v1": "detector_v1", "baseline_megatron": "Megatron-strict",
             "baseline_generic": "generic", "baseline_keyword": "keyword",
             "baseline_xfamily": "cross-family(Nova)"}

    # ---- T1: main baseline comparison ----
    hdr = ["detector", "recall@1", "recall@2", "recall@5", "regfix@2", "benign_FPR", "FPR_lo", "FPR_hi", "$/item"]
    rows = []
    for d in ["detector_v1", "baseline_megatron", "baseline_generic", "baseline_keyword", "baseline_xfamily"]:
        if d not in det:
            continue
        m = det[d]
        fp, n, (lo, hi) = m["fpr"]["_weighted_total"]
        preds = [json.loads(l) for l in (BASE / "predictions" / d / f"{FULL}.jsonl").read_text().splitlines()]
        upc = sum(r.get("cost", 0) for r in preds) / (len(preds) or 1)
        rows.append([NAMES[d], _rrate(m["overall"][1]), _rrate(m["overall"][2]), _rrate(m["overall"][5]),
                     _rrate(m["perkind2"].get("regression-fix", (0, 0))), round(100 * fp / n, 1) if n else 0,
                     round(100 * lo, 1), round(100 * hi, 1), round(upc, 3)])
    _write_csv(TABLES / "T1_baseline_comparison.csv", hdr, rows)
    (TABLES / "T1_baseline_comparison.md").write_text(_md_table(hdr, rows) + "\n")

    # ---- T2: ablation ----
    hdr2 = ["variant", "overall@2", "regfix@2", "benign_FPR"]
    rows2 = []
    for d in ["detector_v1", "ablate_leg1", "ablate_leg2", "ablate_leg3", "ablate_adversarial"]:
        if d not in abl:
            continue
        m = abl[d]
        fp, n, _ = m["fpr"]["_weighted_total"]
        rows2.append([d, _rrate(m["overall"][2]), _rrate(m["perkind2"].get("regression-fix", (0, 0))),
                      round(100 * fp / n, 1) if n else 0])
    _write_csv(TABLES / "T2_ablation.csv", hdr2, rows2)
    (TABLES / "T2_ablation.md").write_text(_md_table(hdr2, rows2) + "\n")

    # ---- T3: per-taxonomy (detector_v1) ----
    m = det["detector_v1"]
    hdr3 = ["category", "detector_v1_recall@2"]
    rows3 = [[c, _rrate(m["percat2"][c])] for c in sorted(m["percat2"], key=lambda k: -m["percat2"][k][1])]
    _write_csv(TABLES / "T3_per_taxonomy.csv", hdr3, rows3)
    (TABLES / "T3_per_taxonomy.md").write_text(_md_table(hdr3, rows3) + "\n")

    # ---- T4: per-kind (all detectors) ----
    kinds = ["regression-fix", "optimization", "config-default-change", "perf-infra-or-test", "not-perf"]
    hdr4 = ["detector"] + kinds
    rows4 = []
    for d in ["detector_v1", "baseline_megatron", "baseline_generic", "baseline_keyword", "baseline_xfamily"]:
        if d not in det:
            continue
        rows4.append([NAMES[d]] + [_rrate(det[d]["perkind2"].get(k, (0, 0))) for k in kinds])
    _write_csv(TABLES / "T4_per_kind.csv", hdr4, rows4)
    (TABLES / "T4_per_kind.md").write_text(_md_table(hdr4, rows4) + "\n")

    # ---- T5: stratified FPR (detector_v1) ----
    hdr5 = ["negative_type", "FP", "N", "rate%", "CI_lo", "CI_hi"]
    rows5 = []
    for nt in sorted(m["fpr"], key=lambda k: (k == "_weighted_total", k)):
        fp, n, (lo, hi) = m["fpr"][nt]
        rows5.append([nt, fp, n, round(100 * fp / n, 1) if n else 0, round(100 * lo, 1), round(100 * hi, 1)])
    _write_csv(TABLES / "T5_stratified_fpr.csv", hdr5, rows5)
    (TABLES / "T5_stratified_fpr.md").write_text(_md_table(hdr5, rows5) + "\n")

    # ---- figures data ----
    # recall@budget curve (detector_v1)
    _write_csv(FIGS / "recall_at_budget.csv", ["budget", "detector_v1_recall%"],
               [[b, _rrate(det["detector_v1"]["overall"][b])] for b in (1, 2, 5)])
    # adversarial recall-FPR (from ablation)
    if "ablate_adversarial" in abl:
        aon, aoff = abl["detector_v1"], abl["ablate_adversarial"]
        fon, non, _ = aon["fpr"]["_weighted_total"]
        foff, noff, _ = aoff["fpr"]["_weighted_total"]
        _write_csv(FIGS / "adversarial_before_after.csv", ["config", "recall@2", "benign_FPR"],
                   [["off", _rrate(aoff["overall"][2]), round(100 * foff / noff, 1) if noff else 0],
                    ["on", _rrate(aon["overall"][2]), round(100 * fon / non, 1) if non else 0]])
    # taxonomy distribution (dataset)
    import re as _re
    l2c = {}
    cur = None
    for ln in (BASE / "taxonomy" / "taxonomy.yaml").read_text().splitlines():
        mm = _re.match(r"^  - id: (\S+)", ln)
        if mm:
            cur = mm.group(1)
        lm = _re.match(r"^      - id: (\S+)", ln)
        if lm:
            l2c[lm.group(1)] = cur
    from collections import Counter
    catcount = Counter(l2c.get(c.get("taxonomy_label"), "?") for c in cards.values() if c.get("is_perf_related"))
    _write_csv(FIGS / "taxonomy_distribution.csv", ["category", "n_cards"],
               sorted(catcount.items(), key=lambda x: -x[1]))
    # detectability distribution
    detc = Counter(c.get("static_detectability") for c in cards.values() if c.get("is_perf_related"))
    _write_csv(FIGS / "detectability_distribution.csv", ["static_detectability", "n_cards"],
               sorted(detc.items(), key=lambda x: -x[1]))
    # funnel counts
    _write_csv(FIGS / "funnel_counts.csv", ["layer", "count"],
               [["perf_cards", 5016], ["regression_pairs", 622], ["pairs_A_tier", 502],
                ["pairs_B_tier", 120], ["negatives", 29703], ["test_frozen_pos", 818],
                ["test_frozen_pairs", 116], ["test_frozen_neg", 954]])

    # cleanup tmp
    import shutil
    if tmp.exists():
        shutil.rmtree(tmp)

    print(f"wrote {len(list(TABLES.glob('*.csv')))} tables + {len(list(FIGS.glob('*.csv')))} figure-data files")


if __name__ == "__main__":
    main()
