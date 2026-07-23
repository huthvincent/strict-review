"""RUN3 Stage 4.4/4.5 — detector_v2 dev report.

Scores detector_v2 on dev_tune with the §4.1 run matrix:
  - OLD framing (primary, per Stage-1 decision): dev_tune original views.
  - CORRECTED framing (appendix): swap the converted regfix cases (24) for their
    inducing-side verdicts (predictions/detector_v2/dev_tune_inducing.jsonl).

Success gates (§4.4, OLD framing, vs RUN2 same-view same-judge):
  overall@2 ≥ 25.3% · pair@2 ≥ 13.3% · weighted FPR ≤ 15% (research gate) / 10% (product line).
Also: stratified FPR (esp hard-negative) + gated-positive count & leaf distribution.
Collision dedup (§4.1): converted-case vs pair colliding on (repo, inducing_sha) counted once.

judge = eval_judge.v1, cache-backed (writes to main cache — these are DEV items, allowed).
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mpb_common import BASE, CASES, PAIRS, REPORTS, read_jsonl, now_iso  # noqa: E402
import eval_metrics as M  # noqa: E402

SPLIT = "dev_tune"
RUN2_REF = {"v1_frozen": {"overall2": 21.3, "pair2": 13.3, "fpr": 12.0},
            "v1_best_ablate_adv": {"overall2": 25.3, "pair2": 10.0, "fpr": 19.0}}
GATE = {"overall2": 25.3, "pair2": 13.3, "fpr_research": 15.0, "fpr_product": 10.0}


def _load(det, name):
    p = BASE / "predictions" / det / f"{name}.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def _wilson(k, n):
    return M.wilson(k, n)


def main():
    cards = {c["case_id"]: c for c in read_jsonl(CASES / "cards_final_v011.jsonl")}
    pairs = {p["pair_id"]: p for p in read_jsonl(PAIRS / "regression_pairs_v011.jsonl")}
    preds = _load("detector_v2", SPLIT)
    induc = {r["item_id"]: r for r in _load("detector_v2", "dev_tune_inducing")}  # caseind:<case_id>

    # ground truth + judge each positive/pair once
    def gt(item_id, kind):
        if kind == "pair":
            pr = pairs[item_id.split(":", 1)[1]]; c = cards.get(pr.get("case_id"), {})
            return {"symptom": pr.get("symptom"), "mechanism": c.get("mechanism", ""), "evidence": c.get("evidence", "")}
        cid = item_id.split(":", 1)[1]
        c = cards.get(cid, {})
        return {"symptom": c.get("symptom"), "mechanism": c.get("mechanism", ""), "evidence": c.get("evidence", "")}

    cost = [0.0]

    def hit2(rec, item_id, kind):
        j, c = M._opus_judge(gt(item_id, kind), rec["findings"])
        cost[0] += c
        if not j.get("hit"):
            return False
        top = M.topk_severe(rec["findings"], 2)
        idx = j.get("hit_finding_index")
        if idx is None or idx >= len(rec["findings"]):
            return len(top) > 0
        return id(rec["findings"][idx]) in {id(f) for f in top}

    # classify items
    pos = [r for r in preds if r["kind"] in ("case", "pair")
           and (r["kind"] == "pair" or cards.get(r["item_id"].split(":", 1)[1], {}).get("is_perf_related"))]
    negs = [r for r in preds if r["kind"] == "neg"]

    # ---- OLD framing recall@2 ----
    def kind_of(r):
        if r["kind"] == "pair":
            return "pair"
        return cards.get(r["item_id"].split(":", 1)[1], {}).get("kind", "?")
    hits_old = {}
    for r in pos:
        hits_old[r["item_id"]] = hit2(r, r["item_id"], r["kind"])
    overall_old = (sum(hits_old.values()), len(pos))
    pair_items = [r for r in pos if r["kind"] == "pair"]
    pair_old = (sum(hits_old[r["item_id"]] for r in pair_items), len(pair_items))
    # by detectability
    def det_of(r):
        if r["kind"] == "pair":
            return "pair"
        return cards.get(r["item_id"].split(":", 1)[1], {}).get("static_detectability", "?")
    strata = defaultdict(lambda: [0, 0])
    for r in pos:
        d = det_of(r); strata[d][1] += 1; strata[d][0] += 1 if hits_old[r["item_id"]] else 0

    # ---- CORRECTED framing: swap converted regfix cases for inducing verdicts ----
    conv_ids = set(cid.split(":", 1)[1] for cid in
                   (r["item_id"] for r in preds if r["kind"] == "case")
                   if f"caseind:{cid.split(':',1)[1]}" in induc)
    # build corrected hit map: for converted cases, use inducing verdict; else old
    hits_corr = dict(hits_old)
    n_swapped = 0
    for r in pos:
        if r["kind"] == "case":
            cid = r["item_id"].split(":", 1)[1]
            ind_rec = induc.get(f"caseind:{cid}")
            if ind_rec:
                hits_corr[r["item_id"]] = hit2(ind_rec, r["item_id"], "case")
                n_swapped += 1
    overall_corr = (sum(hits_corr.values()), len(pos))

    # ---- collision dedup (§4.1): converted-case vs pair on (repo, inducing_sha) ----
    # a converted case and a pair may refer to the same regression → count once
    pair_by_key = {}
    for r in pair_items:
        pr = pairs[r["item_id"].split(":", 1)[1]]
        pair_by_key[(pr["repo"], pr.get("inducing_sha"))] = r["item_id"]
    collisions = 0
    for cid in conv_ids:
        c = cards.get(cid, {})
        # find this case's inducing sha via pairs
        pr = next((p for p in pairs.values() if p.get("case_id") == cid), None)
        if pr and (pr["repo"], pr.get("inducing_sha")) in pair_by_key:
            collisions += 1
    # dedup sensitivity: corrected overall counting each colliding regression once
    overall_corr_dedup = (overall_corr[0], overall_corr[1] - collisions) if collisions else overall_corr

    # ---- FPR (stratified) ----
    from mpb_common import NEGATIVES
    negmeta = {n["case_id"]: n for n in read_jsonl(NEGATIVES / "negatives_v011.jsonl")}
    def is_fp(r):
        return any(f.get("severity") in ("critical", "important") and f.get("confidence", 0) >= 0.5 for f in r["findings"])
    by_nt = defaultdict(lambda: [0, 0])
    for r in negs:
        nt = negmeta.get(r["item_id"].split(":", 1)[1], {}).get("negative_type", "?")
        by_nt[nt][1] += 1; by_nt[nt][0] += 1 if is_fp(r) else 0
    tot_fp = sum(1 for r in negs if is_fp(r)); tot_n = len(negs)

    # ---- gated positives ----
    gated_pos = [r for r in pos if r.get("gate")]
    gated_leaf = Counter(cards.get(r["item_id"].split(":", 1)[1], {}).get("taxonomy_label") for r in gated_pos if r["kind"] == "case")

    def pct(t):
        h, n = t; return 100 * h / n if n else 0

    # ---- gate verdicts ----
    ov = pct(overall_old); pr2 = pct(pair_old); fpr = 100 * tot_fp / tot_n if tot_n else 0
    g_ov = ov >= GATE["overall2"]; g_pr = pr2 >= GATE["pair2"]; g_fpr = fpr <= GATE["fpr_research"]

    L = ["# detector_v2 — dev 全量评测 (RUN3 Stage 4)", "",
         f"- generated: {now_iso()} · judge {M.OPUS} (eval_judge.v1, κ 锚检 0.902) · **旧口径为主**（Stage 1 决定）",
         f"- dev_tune {len(preds)} 项（{len(pos)} 正/pairs, {len(negs)} 负）· judge 花费 ~${cost[0]:.2f}",
         f"- 参照系（RUN2 同视图同 judge, detector_v1_dev.md）：v1 冻结 overall@2 {RUN2_REF['v1_frozen']['overall2']}% / "
         f"pair {RUN2_REF['v1_frozen']['pair2']}% / FPR {RUN2_REF['v1_frozen']['fpr']}%；"
         f"v1 最强 ablate_adv overall@2 {RUN2_REF['v1_best_ablate_adv']['overall2']}% / pair {RUN2_REF['v1_best_ablate_adv']['pair2']}% / FPR {RUN2_REF['v1_best_ablate_adv']['fpr']}%",
         "",
         "## 主表（旧口径，与 RUN2 同口径硬对比）", "",
         "| 指标 | v2 | v1 冻结 | v1 最强(ablate_adv) | 成功判据 | 达标? |",
         "|---|---|---|---|---|:--:|",
         f"| **overall recall@2** | **{overall_old[0]}/{overall_old[1]} ({ov:.1f}%)** | 21.3% | 25.3% | ≥25.3% | {'✓' if g_ov else '✗'} |",
         f"| **pair@2** | **{pair_old[0]}/{pair_old[1]} ({pr2:.1f}%)** | 13.3% | 10.0% | ≥13.3% | {'✓' if g_pr else '✗'} |",
         f"| **weighted FPR** | **{tot_fp}/{tot_n} ({fpr:.1f}%)** | 12.0% | 19.0% | ≤15% (研究门) / ≤10% (产品线) | {'✓' if g_fpr else '✗'}{' / '+('✓' if fpr<=10 else '✗')+'(产品)'} |",
         "",
         "### per static_detectability recall@2 (旧口径)", "",
         "| 层 | recall@2 |", "|---|---|"]
    for d in ["high", "medium", "low", "pair"]:
        if d in strata:
            L.append(f"| {d} | {strata[d][0]}/{strata[d][1]} ({pct(tuple(strata[d])):.1f}%) |")
    L += ["", "## 修正口径节（附表，Stage 1 判定不作主叙事；converted 双跑）", "",
          f"- 转换 regfix case 数（dev_tune∩converted）: **{n_swapped}**（引入侧视图替换）",
          f"- **修正口径 overall@2 = {overall_corr[0]}/{overall_corr[1]} ({pct(overall_corr):.1f}%)**（不得与 25.3% 直接比较）",
          f"- converted-case × pair 在 (repo, inducing_sha) 碰撞数: **{collisions}**"
          + (f" → 去重敏感性 overall@2 = {overall_corr_dedup[0]}/{overall_corr_dedup[1]} ({pct(overall_corr_dedup):.1f}%)" if collisions else "（无碰撞）"),
          "", "## 分层 benign FPR", "", "| negative_type | FP/N | rate | Wilson95 |", "|---|---|--:|---|"]
    for nt in sorted(by_nt):
        fp, n = by_nt[nt]; lo, hi = _wilson(fp, n)
        L.append(f"| {nt} | {fp}/{n} | {100*fp/n if n else 0:.1f}% | [{100*lo:.1f},{100*hi:.1f}] |")
    lo, hi = _wilson(tot_fp, tot_n)
    L.append(f"| **加权总计** | {tot_fp}/{tot_n} | {fpr:.1f}% | [{100*lo:.1f},{100*hi:.1f}] |")
    L += ["", "## 画像门直接判 no_issue 的正样本", "",
          f"- 数量: **{len(gated_pos)}**" + (f" → 叶分布: {dict(gated_leaf)}" if gated_pos else "（无，✓）")]
    if gated_pos:
        for r in gated_pos:
            L.append(f"  - `{r['item_id']}`")
    L += ["", "## 成功判据宣判（旧口径）", "",
          f"- overall@2 {ov:.1f}% {'≥' if g_ov else '<'} 25.3% → **{'达标' if g_ov else '未达标'}**",
          f"- pair@2 {pr2:.1f}% {'≥' if g_pr else '<'} 13.3% → **{'达标' if g_pr else '未达标'}**",
          f"- weighted FPR {fpr:.1f}% {'≤' if g_fpr else '>'} 15% → **{'达标' if g_fpr else '未达标'}**（产品红线 10%: {'达标' if fpr<=10 else '未达标'}）",
          f"- **总判定: {'三项全达标 ✓' if (g_ov and g_pr and g_fpr) else '未全达标 — 如实报告，按消融归因，不放宽判据'}**"]
    (REPORTS / "detector_v2_dev.md").write_text("\n".join(L) + "\n")
    # machine summary
    (REPORTS / "detector_v2_summary.json").write_text(json.dumps({
        "overall2_old": overall_old, "pair2_old": pair_old, "fpr": [tot_fp, tot_n],
        "overall2_corrected": overall_corr, "n_swapped": n_swapped, "collisions": collisions,
        "gates": {"overall2": g_ov, "pair2": g_pr, "fpr15": g_fpr, "fpr10": fpr <= 10},
        "strata": {k: v for k, v in strata.items()}, "judge_cost": round(cost[0], 2),
        "gated_positives": len(gated_pos)}, ensure_ascii=False, indent=2))
    print("\n".join(L))


if __name__ == "__main__":
    main()
