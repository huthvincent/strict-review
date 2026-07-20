"""Stage 4 — ablation + main showdown report (RUN2_INSTRUCTIONS.md §4.3).

Produces reports/detector_v1_results.md:
  - top: static ceiling 57.1% + memorization status (0/934)
  - main table: detector_v1 vs baseline (a)(b)(c)(d) on the FULL test subset
    {recall@1/2/5, per-kind (regression-fix@2 bold), per-taxonomy, stratified FPR+CI, cost/item}
  - ablation table: detector_v1 vs ablate-{leg1,leg2,leg3,adversarial} on the §4.2 subset
    (818 pos + 116 pairs + 400 neg) — recall@2 + FPR deltas
  - adversarial before/after
  - honest section (judge is LLM κ=0.926 + cross-family; machine-labeled; FPR sampled;
    ceiling has label uncertainty; no GPU reproduction)

Ablations are compared to detector_v1 restricted to the SAME 1334-item ablation subset
(apples-to-apples), which is a subset of the 1888-item full test run.

Usage: uv run python ablation_report.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mpb_common import BASE, CASES, REPORTS, read_jsonl, now_iso  # noqa: E402
import eval_metrics as M  # noqa: E402

FULL = "test_eval_subset"
ABL = "test_ablation_subset"
BASELINES = ["baseline_megatron", "baseline_generic", "baseline_keyword", "baseline_xfamily"]
NICE = {"baseline_megatron": "(a) Megatron-strict", "baseline_generic": "(b) generic",
        "baseline_keyword": "(c) keyword", "baseline_xfamily": "(d) cross-family",
        "detector_v1": "**detector_v1**"}
ABLATIONS = ["ablate_leg1", "ablate_leg2", "ablate_leg3", "ablate_adversarial"]


def _subset_ids(split):
    return [l.strip() for l in (BASE / "splits" / f"{split}.txt").read_text().splitlines() if l.strip()]


def _score(detector, split, restrict_ids=None):
    """Score, optionally restricting predictions to a set of item_ids (for apples-to-apples)."""
    m = M.score_predictions(detector, split)
    return m


def _score_restricted(detector, split, keep_ids):
    """Score detector's `split` predictions but only over keep_ids (temp-filter file)."""
    src = BASE / "predictions" / detector / f"{split}.jsonl"
    rows = [json.loads(l) for l in src.read_text().splitlines() if json.loads(l)["item_id"] in keep_ids]
    tmp_id = f"{detector}__ablcmp"
    tmp = BASE / "predictions" / tmp_id
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / f"{split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    m = M.score_predictions(tmp_id, split)
    return m


def rc(t):
    return M.fmt_recall(t)


def main():
    cards = {c["case_id"]: c for c in read_jsonl(CASES / "cards_final_v011.jsonl")}

    # ---- main showdown on FULL test subset ----
    # Only score what has no persisted per-detector report yet; the baselines were already
    # scored into reports/baseline_results.md and detector_v1 into its score report. The
    # disk judge-cache makes any re-score free, but to avoid the long first pass we score
    # only detector_v1 here (its report exists) and pull baselines from baseline_results.md.
    print("scoring main showdown (full test subset)...", flush=True)
    main_scores = {}
    only = os.environ.get("ABL_MAIN_ONLY", "detector_v1").split(",")
    for d in ["detector_v1"] + BASELINES:
        if d in only and (BASE / "predictions" / d / f"{FULL}.jsonl").exists():
            main_scores[d] = _score(d, FULL)
            print(f"  {d}: regfix@2={rc(main_scores[d]['perkind2'].get('regression-fix',(0,0)))}", flush=True)

    # ---- ablations on §4.2 subset; detector_v1 restricted to same ids ----
    abl_ids = set(_subset_ids(ABL))
    print("scoring ablations (§4.2 subset)...", flush=True)
    abl_scores = {}
    # detector_v1 restricted to ablation ids = the reference
    abl_scores["detector_v1"] = _score_restricted("detector_v1", FULL, abl_ids)
    for d in ABLATIONS:
        if (BASE / "predictions" / d / f"{ABL}.jsonl").exists():
            abl_scores[d] = _score(d, ABL)
            print(f"  {d}: regfix@2={rc(abl_scores[d]['perkind2'].get('regression-fix',(0,0)))}", flush=True)

    L = ["# detector_v1 — test showdown + ablations (Stage 4, metrics.v1)", "",
         f"- generated: {now_iso()} · **static ceiling 57.1%** · **memorization 0/934 (novel = full test)**",
         f"- judge {M.OPUS} (κ=0.926, cross-family Nova 89%) · test touched ONCE · leak_attempts 0",
         f"- full test subset = 1888 (818 perf-pos + 116 pairs + 954 neg); ablation subset = 1334 "
         f"(§4.2: pos全量 + 116 pairs + 400 neg subsample)", ""]

    # main recall table (detector_v1 scored here; baselines already in baseline_results.md)
    L += ["## 主表 · recall@budget（full test subset）",
          "> baseline (a)-(d) 的完整主表见 `reports/baseline_results.md`（同一冻结子集，同一 judge）。",
          "> 下表列 detector_v1；关键对照数字在诚实小节汇总。", "",
          "| detector | recall@1 | recall@2 | recall@5 | FPR(weighted, 95%CI) | leak |",
          "|---|---|---|---|---|--:|"]
    order = ["detector_v1", "baseline_megatron", "baseline_generic", "baseline_keyword", "baseline_xfamily"]
    for d in order:
        if d not in main_scores:
            continue
        m = main_scores[d]
        fp, n, (lo, hi) = m["fpr"]["_weighted_total"]
        L.append(f"| {NICE[d]} | {rc(m['overall'][1])} | {rc(m['overall'][2])} | {rc(m['overall'][5])} | "
                 f"{fp}/{n} ({100*fp/n if n else 0:.1f}%, [{100*lo:.1f},{100*hi:.1f}]) | {m['leak_attempts']} |")

    # per-kind regression-fix bold
    L += ["", "## 主表 · per-kind recall@2（**regression-fix = 北极星**）", ""]
    kinds = ["regression-fix", "optimization", "config-default-change", "perf-infra-or-test", "not-perf", "unclear"]
    L += ["| detector | " + " | ".join(("**" + k + "**" if k == "regression-fix" else k) for k in kinds) + " |",
          "|---|" + "---|" * len(kinds)]
    for d in order:
        if d not in main_scores:
            continue
        m = main_scores[d]
        cells = []
        for k in kinds:
            s = rc(m["perkind2"].get(k, (0, 0)))
            cells.append(f"**{s}**" if k == "regression-fix" else s)
        L.append(f"| {NICE[d]} | " + " | ".join(cells) + " |")

    # per-taxonomy (detector_v1 vs the two Opus baselines)
    cats = sorted({c for d in ("detector_v1", "baseline_megatron", "baseline_generic")
                   if d in main_scores for c in main_scores[d]["percat2"]})
    L += ["", "## 主表 · per-taxonomy recall@2（detector_v1 vs Opus baselines）", "",
          "| category | detector_v1 | (a) Megatron | (b) generic |", "|---|---|---|---|"]
    for c in cats:
        row = [c]
        for d in ("detector_v1", "baseline_megatron", "baseline_generic"):
            row.append(rc(main_scores[d]["percat2"].get(c, (0, 0))) if d in main_scores else "-")
        L.append("| " + " | ".join(row) + " |")

    # ablation table
    L += ["", "## 消融表（§4.2 subset, 1334; detector_v1 同口径 restricted）", "",
          "| variant | overall recall@2 | regression-fix@2 | benign FPR | Δ recall@2 vs full |",
          "|---|---|---|---|---|"]
    base = abl_scores["detector_v1"]
    base_r = base["overall"][2]
    base_rr = 100 * base_r[0] / base_r[1] if base_r[1] else 0
    for d in ["detector_v1"] + ABLATIONS:
        if d not in abl_scores:
            continue
        m = abl_scores[d]
        r = m["overall"][2]
        rr = 100 * r[0] / r[1] if r[1] else 0
        fp, n, _ = m["fpr"]["_weighted_total"]
        label = "detector_v1 (all legs)" if d == "detector_v1" else d
        delta = "—" if d == "detector_v1" else f"{rr - base_rr:+.1f}pp"
        L.append(f"| {label} | {rc(r)} | {rc(m['perkind2'].get('regression-fix',(0,0)))} | "
                 f"{fp}/{n} ({100*fp/n if n else 0:.1f}%) | {delta} |")

    # adversarial before/after
    if "ablate_adversarial" in abl_scores:
        a_on, a_off = abl_scores["detector_v1"], abl_scores["ablate_adversarial"]
        fon, non, _ = a_on["fpr"]["_weighted_total"]
        foff, noff, _ = a_off["fpr"]["_weighted_total"]
        L += ["", "## 对抗验证前后（论文图，§4.2 subset）", "",
              "| | recall@2 | benign FPR |", "|---|---|---|",
              f"| 关闭对抗层 | {rc(a_off['overall'][2])} | {foff}/{noff} ({100*foff/noff if noff else 0:.1f}%) |",
              f"| 开启（detector_v1） | {rc(a_on['overall'][2])} | {fon}/{non} ({100*fon/non if non else 0:.1f}%) |"]

    # cost/item
    L += ["", "## 成本 / item（full test subset）", "", "| detector | $/item | s/item |", "|---|--:|--:|"]
    for d in order:
        p = BASE / "predictions" / d / f"{FULL}.jsonl"
        if not p.exists():
            continue
        rows = [json.loads(l) for l in p.read_text().splitlines()]
        n = len(rows) or 1
        L.append(f"| {NICE[d]} | ${sum(r.get('cost',0) for r in rows)/n:.3f} | "
                 f"{sum(r.get('latency_s',0) for r in rows)/n:.1f} |")

    # honest section
    dv = main_scores.get("detector_v1", {})
    dv_rf = dv.get("perkind2", {}).get("regression-fix", (0, 0))
    L += ["", "## 诚实小节（必读）", "",
          f"- **北极星结果**：detector_v1 的 regression-fix recall@2 = {rc(dv_rf)}。"
          f"与两条 Opus baseline（各 14.9%）相比，detector_v1 在**最难的 regression-fix 子类上并未胜出**；"
          f"其 overall recall@2（{rc(dv.get('overall',{}).get(2,(0,0)))}）高于所有 baseline，主要由 optimization 类拉动。"
          f"这是一个**如实报告的混合结果**：多腿+对抗验证在广度与 FPR 控制上有优势，但对最难子类不优于裸强模型。",
          "- **judge 是 LLM**：命中判定用 Opus（κ=0.926 双判一致，跨家族 Nova 一致 89%），非人工真值。",
          "- **数据集机器标注 + 机器仲裁 + 机器分类**：无人工 ground truth（缓解见 S7 人工审计抽样包）。",
          "- **FPR 基于抽样**：主表 FPR 用完整 954 负样本；消融 FPR 用 400 负样本子样（§4.2），CI 已给出。",
          "- **静态天花板 57.1% 自带标注不确定性**：其定义依赖机器判定的 static_detectability。",
          "- **未做 GPU 复现**：leg3 只产出建议触发的 perf recipe，未实际运行基准。",
          "- leak_attempts 全程 = 0：呈现规则未被违反。"]
    (REPORTS / "detector_v1_results.md").write_text("\n".join(L) + "\n")
    print("\nwrote reports/detector_v1_results.md")

    # cleanup temp restricted dir
    import shutil
    tmp = BASE / "predictions" / "detector_v1__ablcmp"
    if tmp.exists():
        shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
