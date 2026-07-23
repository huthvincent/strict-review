"""RUN4 v3 Stage 2 report — detector_v2.1 dev eval (run4.v3 scoring).

Recall (metrics.v1): budget-2 severe finding judged HIT, NO conf threshold.
FPR: negative with a severe∧conf_raw≥0.5 finding.
conf_final is sort-only (never scored).

Main table requirements (§2.5 + 统一口径 §v3):
  - overall recall@2, pair@2, weighted FPR (+ stratified FPR, esp hard-negative)
  - source × conf_raw band (<0.5 / ≥0.5) layered recall & FPR
  - route-HIT / deep-HIT two columns + route-HIT leaf==taxonomy_label agreement
  - SENTINEL row: neg severe∧conf_raw<0.5 fraction vs pos same-band; >15pp gap → flag
  - route invariant, gate_predicate positive list, error count (>5% → BLOCKERS)
  - symmetric口径 (severe∧conf_raw≥0.5 both sides) as appendix, not in verdict
  - overlap sensitivity: full-250 vs minus-smoke-overlap (Mac-40 list untransferred → noted)
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mpb_common import BASE, CASES, PAIRS, NEGATIVES, REPORTS, read_jsonl, now_iso  # noqa: E402
import eval_metrics as M  # noqa: E402

SEV = {"critical", "important"}
GATE = {"overall2": 25.3, "pair2": 13.3, "fpr_res": 15.0, "fpr_prod": 10.0}


def main():
    cards = {c["case_id"]: c for c in read_jsonl(CASES / "cards_final_v011.jsonl")}
    pairs = {p["pair_id"]: p for p in read_jsonl(PAIRS / "regression_pairs_v011.jsonl")}
    negmeta = {n["case_id"]: n for n in read_jsonl(NEGATIVES / "negatives_v011.jsonl")}
    rows = [json.loads(l) for l in (BASE / "predictions" / "detector_v2_1" / "dev_tune.jsonl").read_text().splitlines()]
    pos = [r for r in rows if r["kind"] in ("case", "pair")]
    neg = [r for r in rows if r["kind"] == "neg"]
    errors = [r for r in rows if r.get("error")]

    def gt(r):
        if r["kind"] == "pair":
            pr = pairs[r["item_id"].split(":", 1)[1]]; c = cards.get(pr.get("case_id"), {})
            return {"symptom": pr.get("symptom"), "mechanism": c.get("mechanism", ""), "evidence": c.get("evidence", "")}
        c = cards.get(r["item_id"].split(":", 1)[1], {})
        return {"symptom": c.get("symptom"), "mechanism": c.get("mechanism", ""), "evidence": c.get("evidence", "")}

    def severe(r):
        return [f for f in r["findings"] if f.get("severity") in SEV]  # budget-2 already applied; no conf threshold

    cost = [0.0]
    # judge each positive that has a severe finding (recall side)
    hit_info = {}  # item_id -> (hit_bool, hit_finding)
    for r in pos:
        sv = severe(r)
        if not sv:
            hit_info[r["item_id"]] = (False, None)
            continue
        j, c = M._opus_judge(gt(r), sv)
        cost[0] += c
        hit = bool(j.get("hit"))
        idx = j.get("hit_finding_index")
        hf = sv[idx] if (hit and idx is not None and idx < len(sv)) else (sv[0] if hit else None)
        hit_info[r["item_id"]] = (hit, hf)

    def kind_of(r):
        return "pair" if r["kind"] == "pair" else cards.get(r["item_id"].split(":", 1)[1], {}).get("kind", "?")

    # overall recall@2 + pair@2
    overall_h = sum(1 for r in pos if hit_info[r["item_id"]][0]); overall_n = len(pos)
    pair_items = [r for r in pos if r["kind"] == "pair"]
    pair_h = sum(1 for r in pair_items if hit_info[r["item_id"]][0]); pair_n = len(pair_items)

    # route-HIT / deep-HIT split + leaf agreement
    route_hit = deep_hit = 0
    route_leaf_agree = 0; route_hit_for_leaf = 0
    for r in pos:
        hit, hf = hit_info[r["item_id"]]
        if not hit or hf is None:
            continue
        if hf.get("source") == "route":
            route_hit += 1
            # leaf agreement vs true taxonomy_label
            true_leaf = cards.get(r["item_id"].split(":", 1)[1], {}).get("taxonomy_label") if r["kind"] == "case" else \
                cards.get(pairs[r["item_id"].split(":", 1)[1]].get("case_id"), {}).get("taxonomy_label")
            if true_leaf:
                route_hit_for_leaf += 1
                if hf.get("category") == true_leaf:
                    route_leaf_agree += 1
        else:
            deep_hit += 1

    # FPR (severe∧conf_raw≥0.5) + stratified
    def is_fp(r):
        return any(f.get("severity") in SEV and f.get("conf_raw", 0) >= 0.5 for f in r["findings"])
    by_nt = defaultdict(lambda: [0, 0])
    for r in neg:
        nt = negmeta.get(r["item_id"].split(":", 1)[1], {}).get("negative_type", "?")
        by_nt[nt][1] += 1; by_nt[nt][0] += 1 if is_fp(r) else 0
    tot_fp = sum(1 for r in neg if is_fp(r)); tot_n = len(neg)

    # source × conf_raw band layered recall/FPR
    def band(f):
        return "≥0.5" if f.get("conf_raw", 0) >= 0.5 else "<0.5"
    # sentinel: neg severe∧conf_raw<0.5 fraction vs pos same
    def frac_sub05_severe(items):
        n = 0
        for r in items:
            if any(f.get("severity") in SEV and f.get("conf_raw", 0) < 0.5 for f in r["findings"]):
                n += 1
        return n, len(items)
    neg_sub, neg_tot = frac_sub05_severe(neg)
    pos_sub, pos_tot = frac_sub05_severe(pos)
    neg_sub_pct = 100 * neg_sub / neg_tot if neg_tot else 0
    pos_sub_pct = 100 * pos_sub / pos_tot if pos_tot else 0
    sentinel_gap = neg_sub_pct - pos_sub_pct

    # gate_predicate positives + model step0 no_issue positives
    gate_pos = [r for r in pos if r.get("gate_predicate")]
    step0_pos = [r for r in pos if r.get("model_step0_no_issue")]

    ov = 100 * overall_h / overall_n if overall_n else 0
    pr2 = 100 * pair_h / pair_n if pair_n else 0
    fpr = 100 * tot_fp / tot_n if tot_n else 0
    g_ov, g_pr, g_fpr = ov >= GATE["overall2"], pr2 >= GATE["pair2"], fpr <= GATE["fpr_res"]

    lo_f, hi_f = M.wilson(tot_fp, tot_n)
    L = ["# detector_v2.1 — dev 全量评测 (RUN4 v3 Stage 2)", "",
         f"- generated: {now_iso()} · judge {M.OPUS} (eval_judge.v1, 锚检 κ=0.870) · **run4.v3 口径**",
         f"- 召回=budget2 severe 判 HIT（无 conf 门槛，metrics.v1）· FP=severe∧conf_raw≥0.5 · conf_final 仅排序",
         f"- dev_tune {len(rows)} 项（{len(pos)} 正/pairs, {len(neg)} 负）· error {len(errors)} 项 "
         f"({100*len(errors)/len(rows):.1f}%{'⚠️>5% BLOCKERS' if len(errors)/max(1,len(rows))>0.05 else ''}) · judge ~${cost[0]:.2f}",
         "",
         "## 主表（run4.v3，与 v1 同口径硬对比）", "",
         "| 指标 | v2.1 | v1 冻结 | v1 最强 | 判据 | 达标 |",
         "|---|---|---|---|---|:--:|",
         f"| overall recall@2 | **{overall_h}/{overall_n} ({ov:.1f}%)** | 21.3% | 25.3% | ≥25.3% | {'✓' if g_ov else '✗'} |",
         f"| pair@2 | **{pair_h}/{pair_n} ({pr2:.1f}%)** | 13.3% | 10.0% | ≥13.3% | {'✓' if g_pr else '✗'} |",
         f"| weighted FPR | **{tot_fp}/{tot_n} ({fpr:.1f}%)** [{100*lo_f:.1f},{100*hi_f:.1f}] | 12% | 19% | ≤15%/10%产品 | {'✓' if g_fpr else '✗'}/{'✓' if fpr<=10 else '✗'} |",
         "",
         "## route-HIT / deep-HIT 两列（§2.3）", "",
         f"- route-HIT: **{route_hit}** · deep-HIT: **{deep_hit}** （总命中 {overall_h}）",
         f"- route-HIT 叶子与真值 taxonomy_label 一致: **{route_leaf_agree}/{route_hit_for_leaf}** "
         f"({100*route_leaf_agree/route_hit_for_leaf if route_hit_for_leaf else 0:.0f}%)",
         "",
         "## 哨兵行（反博弈，§v3）", "",
         f"- 负样本 severe∧conf_raw<0.5 占比: **{neg_sub}/{neg_tot} ({neg_sub_pct:.1f}%)**",
         f"- 正样本 severe∧conf_raw<0.5 占比: {pos_sub}/{pos_tot} ({pos_sub_pct:.1f}%)",
         f"- 负−正 gap = **{sentinel_gap:+.1f}pp** "
         + ("⚠️ >15pp：**存在压 conf 躲计分嫌疑**，标注" if sentinel_gap > 15 else "（≤15pp，无压分嫌疑）"),
         "",
         "## 分层 benign FPR (severe∧conf_raw≥0.5)", "",
         "| negative_type | FP/N | rate | Wilson95 |", "|---|---|--:|---|"]
    for nt in sorted(by_nt):
        fp, n = by_nt[nt]; lo, hi = M.wilson(fp, n)
        L.append(f"| {nt} | {fp}/{n} | {100*fp/n if n else 0:.1f}% | [{100*lo:.1f},{100*hi:.1f}] |")
    L += [f"| **加权总计** | {tot_fp}/{tot_n} | {fpr:.1f}% | [{100*lo_f:.1f},{100*hi_f:.1f}] |", "",
          "## 画像门谓词 / 模型步骤0 no_issue 的正样本", "",
          f"- gate_predicate=true 的正样本: **{len(gate_pos)}**" + (f" → {[r['item_id'] for r in gate_pos]}" if gate_pos else "（无）"),
          f"- 模型步骤0 判 no_issue 的正样本: **{len(step0_pos)}**" + (f" → {[r['item_id'] for r in step0_pos]}" if step0_pos else "（无）"),
          "",
          "## route 不变量", f"- 违规项: **{sum(1 for r in rows if r.get('route_invariant_violation'))}**（须=0）",
          "",
          "## 成功判据宣判（run4.v3 口径）", "",
          f"- overall@2 {ov:.1f}% {'≥' if g_ov else '<'} 25.3% → **{'达标' if g_ov else '未达标'}**",
          f"- pair@2 {pr2:.1f}% {'≥' if g_pr else '<'} 13.3% → **{'达标' if g_pr else '未达标'}**",
          f"- weighted FPR {fpr:.1f}% {'≤' if g_fpr else '>'} 15% → **{'达标' if g_fpr else '未达标'}**（产品线10%: {'达标' if fpr<=10 else '未达标'}）",
          f"- **总判定: {'三项全达标 ✓' if (g_ov and g_pr and g_fpr) else '未全达标 — 如实报告，不放宽不重跑'}**"]
    (REPORTS / "detector_v2_1_dev.md").write_text("\n".join(L) + "\n")
    (REPORTS / "detector_v2_1_summary.json").write_text(json.dumps({
        "overall2": [overall_h, overall_n], "pair2": [pair_h, pair_n], "fpr": [tot_fp, tot_n],
        "route_hit": route_hit, "deep_hit": deep_hit, "route_leaf_agree": [route_leaf_agree, route_hit_for_leaf],
        "sentinel_gap_pp": round(sentinel_gap, 1), "gates": {"overall2": g_ov, "pair2": g_pr, "fpr15": g_fpr, "fpr10": fpr <= 10},
        "errors": len(errors), "gate_pos": len(gate_pos), "judge_cost": round(cost[0], 2)}, ensure_ascii=False, indent=2))
    print("\n".join(L))


if __name__ == "__main__":
    main()
