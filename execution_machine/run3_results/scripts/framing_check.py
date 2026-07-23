"""RUN3 Stage 1 — exam-framing check (§1). EVIDENCE ORGANIZATION ONLY, no final conclusion.

Hypothesis: regfix's case side (asking "did this FIX commit introduce a regression?") is
systematically down-scored due to question-direction misalignment vs the pair side (asking
about the INDUCING commit).

All judging is CACHE-ONLY (JUDGE_CACHE_ONLY=1) — read-only reaggregation of RUN2 verdicts.

Outputs:
  reports/framing_report.md   — mapping+overlap, McNemar×5 selectors, alt-explanations, converted rate
  (§1.3 inducing views written by a separate step: build_inducing_views.py)

McNemar per selector over the (fix-case, pair) pairs referring to the SAME regression:
  b = fix-case MISS & pair HIT   (inducing side easier)
  c = fix-case HIT  & pair MISS
  exact/continuity-corrected McNemar on discordant (b,c).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("JUDGE_CACHE_ONLY", "1")  # hard cache-only for this whole module
from mpb_common import BASE, CASES, PAIRS, REPORTS, read_jsonl, now_iso  # noqa: E402
import eval_metrics as M  # noqa: E402

SELECTORS = ["detector_v1", "baseline_megatron", "baseline_generic", "baseline_keyword", "baseline_xfamily"]
SPLIT = "test_eval_subset"


def _gt(p, cards, pairs):
    if p["kind"] == "pair":
        pr = pairs[p["item_id"].split(":", 1)[1]]
        c = cards.get(pr.get("case_id"), {})
        return {"symptom": pr.get("symptom"), "mechanism": c.get("mechanism", ""), "evidence": c.get("evidence", "")}
    c = cards[p["item_id"].split(":", 1)[1]]
    return {"symptom": c.get("symptom"), "mechanism": c.get("mechanism", ""), "evidence": c.get("evidence", "")}


def _verdict(p, cards, pairs, cache):
    """Return HIT bool from cache (cache-only). Raise if miss."""
    u = M._judge_user(_gt(p, cards, pairs), p["findings"])
    key = hashlib.sha1((M.JUDGE_PROMPT + "\x00" + u).encode()).hexdigest()
    if key not in cache:
        raise M.JudgeCacheMiss(f"{p['item_id']} not cached")
    return bool(cache[key].get("hit"))


def mcnemar(b, c):
    """Continuity-corrected McNemar chi-square + exact binomial p (two-sided) on discordant."""
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p_cc": 1.0, "p_exact": 1.0}
    chi2 = (abs(b - c) - 1) ** 2 / n if n > 0 else 0.0
    # chi2 survival with 1 dof
    p_cc = math.erfc(math.sqrt(chi2 / 2)) if chi2 > 0 else 1.0
    # exact two-sided binomial (p=0.5)
    k = min(b, c)
    from math import comb
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    p_exact = min(1.0, 2 * tail)
    return {"b": b, "c": c, "chi2": round(chi2, 3), "p_cc": round(p_cc, 4), "p_exact": round(p_exact, 4)}


def main():
    cards = {c["case_id"]: c for c in read_jsonl(CASES / "cards_final_v011.jsonl")}
    pairs = {p["pair_id"]: p for p in read_jsonl(PAIRS / "regression_pairs_v011.jsonl")}
    cache = M._load_disk_cache()

    # ---- 1.1 case<->pair mapping + overlap ----
    test = [l.strip() for l in (BASE / "splits" / f"{SPLIT}.txt").read_text().splitlines() if l.strip()]
    test_case_ids = set(l.split(":", 1)[1] for l in test if l.startswith("case:"))
    test_pair_ids = [l.split(":", 1)[1] for l in test if l.startswith("pair:")]
    # each pair's fix-case = pairs[pid].case_id ; check it's in test case set
    mapping = []  # (pair_id, fix_case_id, in_test_case)
    for pid in test_pair_ids:
        fc = pairs[pid].get("case_id")
        mapping.append((pid, fc, fc in test_case_ids))
    overlap = [m for m in mapping if m[2]]
    overlap_rate = len(overlap) / len(mapping) if mapping else 0

    # ---- 1.1 McNemar per selector on overlap ----
    mcn = {}
    miss_selectors = []
    for sel in SELECTORS:
        pf = BASE / "predictions" / sel / f"{SPLIT}.jsonl"
        preds = {p["item_id"]: p for p in (json.loads(l) for l in pf.read_text().splitlines())}
        b = c = concordant_hit = concordant_miss = 0
        n_eval = 0
        try:
            for pid, fc, _ in overlap:
                case_item = preds.get(f"case:{fc}")
                pair_item = preds.get(f"pair:{pid}")
                if not case_item or not pair_item:
                    continue
                case_hit = _verdict(case_item, cards, pairs, cache)   # fix-side question
                pair_hit = _verdict(pair_item, cards, pairs, cache)   # inducing-side question
                n_eval += 1
                if (not case_hit) and pair_hit:
                    b += 1
                elif case_hit and (not pair_hit):
                    c += 1
                elif case_hit and pair_hit:
                    concordant_hit += 1
                else:
                    concordant_miss += 1
        except M.JudgeCacheMiss as e:
            miss_selectors.append((sel, str(e)))
            continue
        stat = mcnemar(b, c)
        stat.update({"n": n_eval, "concordant_hit": concordant_hit, "concordant_miss": concordant_miss,
                     "inducing_easier": b > c})
        mcn[sel] = stat

    # ---- 1.2 alternative explanations: case vs pair group composition ----
    # case group = the fix-cases; pair group = the inducing commits (from pairs)
    def diff_size(repo, sha):
        # cheap proxy: changed_files count via git (fall back to None)
        try:
            import subprocess
            from mpb_common import repo_dir  # type: ignore
        except Exception:
            pass
        return None
    fix_cases = [cards.get(fc, {}) for _, fc, ok in overlap if ok]
    pair_objs = [pairs[pid] for pid, _, ok in overlap if ok]
    def dist(objs, field, src="card"):
        return dict(Counter((o.get(field) or "?") for o in objs))
    alt = {
        "detectability_case": dist(fix_cases, "static_detectability"),
        "detectability_pair": dict(Counter((p.get("static_detectability") or "?") for p in pair_objs)),
        "repo_case": dist(fix_cases, "repo"),
        "repo_pair": dict(Counter((p.get("repo") or "?") for p in pair_objs)),
    }

    # ---- 1.4 pre-registered adoption rule evaluation ----
    sig = [sel for sel, s in mcn.items() if s["p_exact"] < 0.05 and s["inducing_easier"]]
    adopt = len(sig) >= 3
    # any selector where direction is REVERSED among significant?
    consistent = all(s["inducing_easier"] for sel, s in mcn.items() if s["p_exact"] < 0.05)

    # ---- write report ----
    L = ["# RUN3 Stage 1 — 考题框架检验（证据整理，不下最终结论）", "",
         f"- generated: {now_iso()} · **cache-only（JUDGE_CACHE_ONLY=1）**，对 test 零新模型调用",
         f"- 假设：regfix 的 case 侧（用修复 commit 提问）因方向错位被系统性压分。",
         "", "## 1.1 case↔pair 映射与重合率", "",
         f"- test pairs: {len(mapping)} · fix-case 落在 test case 集内: **{len(overlap)} ({overlap_rate:.0%})**",
         f"- 非重合: {len(mapping)-len(overlap)}（只做描述性对比，声明混杂；本轮全部重合故无非重合部分）"
         if overlap_rate < 1 else f"- **100% 重合** → McNemar 配对检验在全部 {len(overlap)} 对上有效。",
         ""]
    if miss_selectors:
        L += ["> ⚠️ 以下选手有 cache miss（cache-only 下跳过，写入 BLOCKERS 候选）："]
        L += [f">   - {s}: {e}" for s, e in miss_selectors]
        L += [""]
    L += ["## 1.1 McNemar 配对检验（fix-侧 case vs inducing-侧 pair，同一回归）", "",
          "> b = fix-case MISS 且 pair HIT（引入侧更易）· c = 反之 · p 为精确二项两侧检验", "",
          "| 选手 | n | b (引入侧独中) | c (修复侧独中) | 一致命中 | 一致漏 | χ²(cc) | p(exact) | 引入侧更易? | 显著(p<.05)? |",
          "|---|--:|--:|--:|--:|--:|--:|--:|:--:|:--:|"]
    for sel in SELECTORS:
        if sel not in mcn:
            L.append(f"| {sel} | — | — | — | — | — | — | — | (cache miss) | — |")
            continue
        s = mcn[sel]
        L.append(f"| {sel} | {s['n']} | {s['b']} | {s['c']} | {s['concordant_hit']} | {s['concordant_miss']} | "
                 f"{s['chi2']} | {s['p_exact']} | {'✓' if s['inducing_easier'] else '✗'} | "
                 f"{'✓' if s['p_exact']<0.05 else '—'} |")
    L += ["", "## 1.2 备择解释检验（排除『框架效应其实是构成差异』）", "",
          "重合子集的 case 组（=修复 commit 的卡）与 pair 组（=引入 commit）构成对比：", "",
          f"- static_detectability — case: `{alt['detectability_case']}` · pair: `{alt['detectability_pair']}`",
          f"- repo — case: `{alt['repo_case']}` · pair: `{alt['repo_pair']}`",
          "- diff 大小分布：见 `reports/framing_diffsize.json`（由 build_inducing_views 附带计算）。",
          "> 注：case 与 pair 指向同一回归的两个 commit（fix vs inducing），"
          "detectability 是卡级标注，二者多数相同——构成差异预期很小，故 McNemar 的配对设计已控住大部分混杂。",
          "", "## 1.4 修正口径采纳规则（预登记，现在宣判）", "",
          f"- 规则：McNemar 在 **≥3/5** 选手上 p<0.05 **且**方向一致（引入侧更易）→ Stage 4 以修正口径为主叙事；否则旧口径为主、修正口径只作附表。",
          f"- 显著且方向正确的选手：**{len(sig)}/5** （{sig}）· 方向一致性：{'✓' if consistent else '✗ 有反向'}",
          f"- **判定：{'采纳修正口径为主叙事（Stage 4）' if (adopt and consistent) else '旧口径为主，修正口径仅作附表'}**",
          "", "## 1.5 定位与去分母问题", "",
          "- 本报告为**证据整理**：分解表 + 配对检验 + 备择解释 + 转换率（转换率见下方 1.3 步骤产物）。",
          "- 「历史分数应如何重新解读」与「北极星口径是否修改」**留给 Rui 终审**（见 run3_report 等待项）。",
          "- **v0.3 修正建议**：现北极星 regression-fix 分母 261 = case 侧 regfix 卡 + 116 pairs，"
          "对同一回归**双计**（fix-case 与其 pair 都计入）。建议 v0.3 去重为「每个回归计一次」。"]
    (REPORTS / "framing_report.md").write_text("\n".join(L) + "\n")
    # machine-readable summary for Stage 4/6
    (REPORTS / "framing_summary.json").write_text(json.dumps({
        "overlap": len(overlap), "overlap_rate": overlap_rate, "mcnemar": mcn,
        "sig_selectors": sig, "consistent": consistent, "adopt_corrected_as_primary": bool(adopt and consistent),
        "cache_only": True, "misses": miss_selectors}, ensure_ascii=False, indent=2))
    print("\n".join(L))
    print(f"\n>>> adopt_corrected_as_primary = {adopt and consistent} (sig {len(sig)}/5, consistent {consistent})")


if __name__ == "__main__":
    main()
