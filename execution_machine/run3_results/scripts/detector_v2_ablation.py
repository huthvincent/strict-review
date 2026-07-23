"""RUN3 Stage 4.2/4.3 — κ recheck + cross-family + 2 ablations with Wilson CI.

§4.2 κ recheck: 100 items stratified by pos/neg × HIT/MISS, SECOND judgement bypasses cache
   (user prompt gets a one-time nonce; result NOT written to main cache). κ≥0.7 to continue.
§4.2 cross-family: 50 items (over-sampled to v2 HIT) judged by Nova Pro; agreement reported.
§4.3 ablations: v2 vs v2−handbook vs v2−tools on the 80-item subset. recall@2 + Wilson 95% CI.
   CI overlap → "不可区分". "归因边界" note: gate + large-commit-decomp contributions not isolated.

Writes reports/detector_v2_ablation.md (+ merges κ/xfam into detector_v2_dev via summary).
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mpb_common import BASE, CASES, PAIRS, REPORTS, read_jsonl, now_iso  # noqa: E402
import eval_metrics as M  # noqa: E402

SEED = 20260720


def _gt(item_id, kind, cards, pairs):
    if kind == "pair":
        pr = pairs[item_id.split(":", 1)[1]]; c = cards.get(pr.get("case_id"), {})
        return {"symptom": pr.get("symptom"), "mechanism": c.get("mechanism", ""), "evidence": c.get("evidence", "")}
    c = cards.get(item_id.split(":", 1)[1], {})
    return {"symptom": c.get("symptom"), "mechanism": c.get("mechanism", ""), "evidence": c.get("evidence", "")}


def _judge_nocache(gt, findings, nonce):
    """Second judgement bypassing cache (nonce in user prompt); NOT written back."""
    u = M._judge_user(gt, findings) + f"\n\n[recheck-nonce: {nonce}]"
    with M._cl().messages.stream(model=M.OPUS, max_tokens=600, thinking={"type": "disabled"},
                                 system=M._judge_system(), messages=[{"role": "user", "content": u}],
                                 tools=[M.JUDGE_EMIT], tool_choice={"type": "tool", "name": "emit_judgement"}) as st:
        r = st.get_final_message()
    return next((x.input for x in r.content if x.type == "tool_use"), {}) or {}


def _hit2(rec, item_id, kind, cards, pairs):
    j, _ = M._opus_judge(_gt(item_id, kind, cards, pairs), rec["findings"])
    if not j.get("hit"):
        return False
    top = M.topk_severe(rec["findings"], 2)
    idx = j.get("hit_finding_index")
    if idx is None or idx >= len(rec["findings"]):
        return len(top) > 0
    return id(rec["findings"][idx]) in {id(f) for f in top}


def score_subset(det, split, cards, pairs, negmeta):
    rows = [json.loads(l) for l in (BASE / "predictions" / det / f"{split}.jsonl").read_text().splitlines()]
    pos = [r for r in rows if r["kind"] in ("case", "pair")
           and (r["kind"] == "pair" or cards.get(r["item_id"].split(":", 1)[1], {}).get("is_perf_related"))]
    negs = [r for r in rows if r["kind"] == "neg"]
    h = sum(1 for r in pos if _hit2(r, r["item_id"], r["kind"], cards, pairs))
    def is_fp(r):
        return any(f.get("severity") in ("critical", "important") and f.get("confidence", 0) >= 0.5 for f in r["findings"])
    fp = sum(1 for r in negs if is_fp(r))
    return {"recall2": (h, len(pos)), "fpr": (fp, len(negs))}


def main():
    cards = {c["case_id"]: c for c in read_jsonl(CASES / "cards_final_v011.jsonl")}
    pairs = {p["pair_id"]: p for p in read_jsonl(PAIRS / "regression_pairs_v011.jsonl")}
    from mpb_common import NEGATIVES
    negmeta = {n["case_id"]: n for n in read_jsonl(NEGATIVES / "negatives_v011.jsonl")}
    rnd = random.Random(SEED)

    # ---- §4.2 κ recheck (100 items, pos/neg × HIT/MISS strata) ----
    dev = [json.loads(l) for l in (BASE / "predictions" / "detector_v2" / "dev_tune.jsonl").read_text().splitlines()]
    strata = defaultdict(list)
    for r in dev:
        if r["kind"] in ("case", "pair") and (r["kind"] == "pair" or cards.get(r["item_id"].split(":", 1)[1], {}).get("is_perf_related")):
            hit = _hit2(r, r["item_id"], r["kind"], cards, pairs)
            strata[("pos", hit)].append(r)
        elif r["kind"] == "neg":
            fp = any(f.get("severity") in ("critical", "important") and f.get("confidence", 0) >= 0.5 for f in r["findings"])
            strata[("neg", fp)].append(r)
    kappa_sample = []
    per = 25
    for key, items in strata.items():
        kappa_sample += rnd.sample(items, min(per, len(items)))
    kappa_sample = kappa_sample[:100]
    a1, a2 = [], []
    for i, r in enumerate(kappa_sample):
        j1, _ = M._opus_judge(_gt(r["item_id"], r["kind"], cards, pairs), r["findings"])
        j2 = _judge_nocache(_gt(r["item_id"], r["kind"], cards, pairs), r["findings"], f"n{i}")
        a1.append(str(bool(j1.get("hit")))); a2.append(str(bool(j2.get("hit"))))
    kappa = M.cohens_kappa(a1, a2)
    agree = sum(1 for x, y in zip(a1, a2) if x == y) / len(a1) if a1 else 0

    # ---- §4.2 cross-family (50 items, over-sample v2 HITs) ----
    hits = strata[("pos", True)]
    others = strata[("pos", False)] + strata[("neg", False)] + strata[("neg", True)]
    xf_items = (rnd.sample(hits, min(30, len(hits))) + rnd.sample(others, min(20, len(others))))[:50]
    xf = []
    for r in xf_items:
        gt = _gt(r["item_id"], r["kind"], cards, pairs)
        jo, _ = M._opus_judge(gt, r["findings"])
        try:
            jx = M._nonanthropic_judge(gt, r["findings"], "us.amazon.nova-pro-v1:0")
        except Exception:
            continue
        xf.append((bool(jo.get("hit")), bool(jx.get("hit"))))
    xf_agree = sum(1 for x, y in xf if x == y) / len(xf) if xf else 0

    # ---- §4.3 ablations ----
    v2 = score_subset("detector_v2", "v2_ablation_subset", cards, pairs, negmeta) \
        if (BASE / "predictions" / "detector_v2" / "v2_ablation_subset.jsonl").exists() else None
    # v2 on the ablation subset may not exist as a separate run — compute from dev_tune preds restricted
    if v2 is None:
        sub_ids = set(l.strip() for l in (BASE / "splits" / "v2_ablation_subset.txt").read_text().splitlines() if l.strip())
        rows = [json.loads(l) for l in (BASE / "predictions" / "detector_v2" / "dev_tune.jsonl").read_text().splitlines()
                if json.loads(l)["item_id"] in sub_ids]
        tmp = BASE / "predictions" / "detector_v2__ablref"; tmp.mkdir(parents=True, exist_ok=True)
        (tmp / "v2_ablation_subset.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        v2 = score_subset("detector_v2__ablref", "v2_ablation_subset", cards, pairs, negmeta)
    abl_hb = score_subset("v2_ablate_handbook", "v2_ablation_subset", cards, pairs, negmeta)
    abl_tl = score_subset("v2_ablate_tools", "v2_ablation_subset", cards, pairs, negmeta)

    def ci(t):
        k, n = t; lo, hi = M.wilson(k, n); return (100 * k / n if n else 0, 100 * lo, 100 * hi)

    def overlap(a, b):
        _, alo, ahi = ci(a["recall2"]); _, blo, bhi = ci(b["recall2"])
        return not (ahi < blo or bhi < alo)

    L = ["# detector_v2 — κ/跨家族/消融 (RUN3 Stage 4.2/4.3)", "",
         f"- generated: {now_iso()} · judge {M.OPUS}", "",
         "## §4.2 κ 复检（100 项，pos/neg×HIT/MISS 分层，第二次判分绕缓存加 nonce）",
         f"- 样本 {len(a1)} · 一致率 {agree:.1%} · **Cohen κ = {kappa:.3f}** "
         + ("✓ ≥0.7 继续" if (kappa or 0) >= 0.7 else "✗ <0.7 → 停等 Rui"),
         "", "## §4.2 跨家族抽检（Nova Pro，50 项过采 v2 HIT）",
         f"- 样本 {len(xf)} · 与 Opus-judge 一致率 **{xf_agree:.1%}**"
         + ("（Opus 全链路自我偏好对照）" if xf_agree >= 0.8 else " ⚠️ 分歧较大，报告标注"),
         "", "## §4.3 预登记消融（80 项子集：50 正+30 负，剔除冒烟重叠；Wilson 95% CI）", "",
         "| 变体 | recall@2 | 95% CI | benign FPR |", "|---|---|---|---|"]
    for name, m in [("detector_v2 (full)", v2), ("v2 − handbook", abl_hb), ("v2 − tools(父快照)", abl_tl)]:
        r, lo, hi = ci(m["recall2"]); fp = m["fpr"]
        L.append(f"| {name} | {m['recall2'][0]}/{m['recall2'][1]} ({r:.1f}%) | [{lo:.1f},{hi:.1f}] | {fp[0]}/{fp[1]} ({100*fp[0]/fp[1] if fp[1] else 0:.1f}%) |")
    L += ["",
          f"- v2 vs −handbook: {'CI 重叠 → 不可区分' if overlap(v2, abl_hb) else '手册有可区分贡献'}",
          f"- v2 vs −tools: {'CI 重叠 → 不可区分' if overlap(v2, abl_tl) else '父快照工具有可区分贡献'}",
          "", "## 归因边界（§4.3 要求）",
          "- **画像门**与**大 commit 分解**的贡献本轮**未单独隔离**（两项消融只切手册与父快照工具）。",
          "- 画像门在 dev_tune 直接判 no_issue 的正样本数见 `reports/detector_v2_dev.md`（gated-positives 节）。"]
    (REPORTS / "detector_v2_ablation.md").write_text("\n".join(L) + "\n")
    (REPORTS / "detector_v2_ablation_summary.json").write_text(json.dumps({
        "kappa": kappa, "kappa_agree": agree, "xfamily_agree": xf_agree, "n_xf": len(xf),
        "v2": v2, "ablate_handbook": abl_hb, "ablate_tools": abl_tl}, ensure_ascii=False, indent=2))
    print("\n".join(L))
    import shutil
    t = BASE / "predictions" / "detector_v2__ablref"
    if t.exists():
        shutil.rmtree(t)


if __name__ == "__main__":
    main()
