"""Stage 7 — human-audit sampling packet (RUN2_INSTRUCTIONS.md §7).

Prepares MATERIAL ONLY (no labels). Deterministic stratified samples:
  A组 · 357 cards  (5,016 perf cards @ 95% conf / 5% margin), stratified by
        repo × kind × static_detectability, covering all 11 categories (≥5 each).
  B组 · 150 pairs  (A-tier 100 / B-tier 50), stratified by repo.
Outputs:
  audit/sample_manifest.json          — method + seed + strata counts
  audit/human_audit_packet.md          — offline-readable, one block per item + blanks
  audit/human_audit_template.jsonl     — prefilled target_id/field/machine_value, verdict blank
  reports/human_audit_packet.md        — sampling method + strata table + agreement/CI formulas
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mpb_common import BASE, CASES, PAIRS, REPORTS, read_jsonl, now_iso  # noqa: E402
import eval_harness as H  # noqa: E402

import random

AUDIT = BASE / "audit"
SEED = 20260719
A_N = 357
B_N = 150
B_A_TIER, B_B_TIER = 100, 50


def _leaf_to_category():
    m, cur = {}, None
    for ln in (BASE / "taxonomy" / "taxonomy.yaml").read_text().splitlines():
        mm = re.match(r"^  - id: (\S+)", ln)
        if mm:
            cur = mm.group(1)
        lm = re.match(r"^      - id: (\S+)", ln)
        if lm:
            m[lm.group(1)] = cur
    return m


def _proportional_strata(items, key_fn, n, rnd):
    """Largest-remainder allocation across strata, then random sample within each."""
    groups = defaultdict(list)
    for it in items:
        groups[key_fn(it)].append(it)
    total = len(items)
    raw = {k: len(v) / total * n for k, v in groups.items()}
    alloc = {k: int(x) for k, x in raw.items()}
    rem = n - sum(alloc.values())
    for k in sorted(groups, key=lambda k: raw[k] - alloc[k], reverse=True)[:rem]:
        alloc[k] += 1
    picked = []
    for k, v in groups.items():
        take = min(alloc.get(k, 0), len(v))
        picked += rnd.sample(sorted(v, key=lambda c: c.get("case_id", c.get("pair_id", ""))), take)
    return picked, {k: alloc.get(k, 0) for k in groups}


def build_A(cards, rnd, l2c):
    perf = [c for c in cards if c.get("is_perf_related")]
    picked, strata = _proportional_strata(
        perf, lambda c: (c["repo"], c.get("kind"), c.get("static_detectability")), A_N, rnd)
    # ensure every category has >=5
    by_cat = defaultdict(list)
    for c in picked:
        by_cat[l2c.get(c.get("taxonomy_label"), "?")].append(c)
    picked_ids = {c["case_id"] for c in picked}
    for cat in set(l2c.values()):
        have = len(by_cat.get(cat, []))
        if have < 5:
            pool = [c for c in perf if l2c.get(c.get("taxonomy_label")) == cat
                    and c["case_id"] not in picked_ids]
            add = rnd.sample(sorted(pool, key=lambda c: c["case_id"]), min(5 - have, len(pool)))
            for c in add:
                picked.append(c); picked_ids.add(c["case_id"]); by_cat[cat].append(c)
    return picked, strata


def build_B(pairs, rnd):
    a_tier = [p for p in pairs if p.get("evidence_tier") == "A"]
    b_tier = [p for p in pairs if p.get("evidence_tier") == "B"]
    pa, sa = _proportional_strata(a_tier, lambda p: p["repo"], min(B_A_TIER, len(a_tier)), rnd)
    pb, sb = _proportional_strata(b_tier, lambda p: p["repo"], min(B_B_TIER, len(b_tier)), rnd)
    return pa + pb, {"A_tier_by_repo": sa, "B_tier_by_repo": sb}


def _diff_hunk(repo, sha, limit=40):
    try:
        view = H.build_view(repo, sha)
        lines = view["diff"].splitlines()
        return "\n".join(lines[:limit]) + ("\n... [truncated] ..." if len(lines) > limit else "")
    except Exception as e:
        return f"[diff unavailable: {str(e)[:80]}]"


def main():
    rnd = random.Random(SEED)
    AUDIT.mkdir(parents=True, exist_ok=True)
    cards = list(read_jsonl(CASES / "cards_final_v011.jsonl"))
    card_by_id = {c["case_id"]: c for c in cards}
    pairs = list(read_jsonl(PAIRS / "regression_pairs_v011.jsonl"))
    l2c = _leaf_to_category()

    A, A_strata = build_A(cards, rnd, l2c)
    B, B_strata = build_B(pairs, rnd)
    print(f"A组 {len(A)} cards · B组 {len(B)} pairs", flush=True)

    # manifest
    manifest = {
        "generated": now_iso(), "seed": SEED,
        "A_group": {"target_n": A_N, "actual_n": len(A),
                    "basis": "5,016 perf cards @ 95% conf / 5% margin",
                    "strata": "repo × kind × static_detectability; +top-up to ≥5 per category",
                    "strata_counts": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in A_strata.items()}},
        "B_group": {"target_n": B_N, "actual_n": len(B),
                    "basis": "A-tier 100 / B-tier 50, stratified by repo", "strata_counts": B_strata},
        "category_coverage": dict(Counter(l2c.get(c.get("taxonomy_label"), "?") for c in A)),
    }
    (AUDIT / "sample_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    # template jsonl (prefilled machine values, verdict blank)
    tmpl = []
    for c in A:
        for field in ("kind", "symptom", "mechanism", "taxonomy_label"):
            tmpl.append({"group": "A", "target_id": c["case_id"], "field": field,
                         "machine_value": c.get(field), "human_value": None, "verdict": None})
    for p in B:
        tmpl.append({"group": "B", "target_id": p["pair_id"], "field": "is_true_regression_pair",
                     "machine_value": True, "human_value": None, "verdict": None})
    (AUDIT / "human_audit_template.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in tmpl))

    # readable packet
    L = ["# Human Audit Packet (Stage 7 — 材料,不含标签)", "",
         f"seed {SEED} · A组 {len(A)} 卡 · B组 {len(B)} 对 · 生成 {now_iso()}",
         "", "> 每项：机器结论 + evidence 引文 + 关键 diff hunk（≤40 行）+ 待填空位。",
         "> 判定：`[ ] 同意  [ ] 不同意 → 正确值: ____  [ ] 无法判断`。建议优先判 B 组。", "",
         "---", "", "## A 组 · 卡片判定", ""]
    for i, c in enumerate(sorted(A, key=lambda c: c["case_id"]), 1):
        L += [f"### A{i}. `{c['case_id']}`  ({c['repo']})", "",
              f"- **机器 kind**: {c.get('kind')}",
              f"- **机器 symptom**: {c.get('symptom')}",
              f"- **机器 mechanism**: {(c.get('mechanism') or '')[:400]}",
              f"- **taxonomy 叶**: {c.get('taxonomy_label')} ({l2c.get(c.get('taxonomy_label'),'?')})",
              f"- **static_detectability**: {c.get('static_detectability')}",
              f"- **evidence 引文**: {(c.get('evidence') or '')[:300]}", "",
              "```diff", _diff_hunk(c["repo"], c["sha"]), "```", "",
              "判定: `[ ] 同意   [ ] 不同意 → 正确值: ____   [ ] 无法判断`", "", "---", ""]
        if i % 20 == 0:
            print(f"  A组 {i}/{len(A)} 渲染", flush=True)
    L += ["## B 组 · 回归对判定", ""]
    for i, p in enumerate(sorted(B, key=lambda p: p["pair_id"]), 1):
        c = card_by_id.get(p.get("case_id"), {})
        L += [f"### B{i}. `{p['pair_id']}`  ({p['repo']}, tier {p.get('evidence_tier')})", "",
              f"- **inducing**: {p.get('inducing_sha','')[:12]}  →  **fix**: {p.get('fix_sha','')[:12]}",
              f"- **机器 symptom**: {p.get('symptom')}",
              f"- **归因原文/来源**: {p.get('evidence_source')}  (szz_votes: {p.get('szz_votes')})",
              f"- **fix commit subject**: {c.get('subject','(n/a)')}", "",
              "关键 diff (fix):", "```diff", _diff_hunk(p["repo"], p.get("fix_sha", "")), "```", "",
              "判定: `[ ] 真回归对   [ ] 非回归对 → 说明: ____   [ ] 无法判断`", "", "---", ""]
        if i % 20 == 0:
            print(f"  B组 {i}/{len(B)} 渲染", flush=True)
    (AUDIT / "human_audit_packet.md").write_text("\n".join(L) + "\n")

    # methodology report
    R = ["# Human Audit Packet — 抽样方法与计算口径 (Stage 7)", "",
         f"- 生成 {now_iso()} · seed {SEED}", "",
         "## 抽样",
         f"- **A 组 {len(A)} 卡**：从 5,016 张 perf 卡按 repo × kind × static_detectability "
         f"比例分层（最大余数法），再对不足 5 项的大类补足至 ≥5，覆盖全部 11 类。",
         f"- **B 组 {len(B)} 对**：A 级 {sum(1 for p in B if p.get('evidence_tier')=='A')} / "
         f"B 级 {sum(1 for p in B if p.get('evidence_tier')=='B')}，按 repo 分层。", "",
         "## 类别覆盖",
         "| category | A组抽样数 |", "|---|--:|"]
    for cat, n in sorted(Counter(l2c.get(c.get("taxonomy_label"), "?") for c in A).items()):
        R.append(f"| {cat} | {n} |")
    R += ["", "## agreement 与 95% CI 计算口径",
          "- 回填 `human_audit_template.jsonl` 后：",
          "  - **per-field agreement** = 同意数 / (同意 + 不同意)，'无法判断' 从分母剔除并单列。",
          "  - **Cohen's κ**（机器 vs 人）：对 kind/taxonomy 这类分类字段，按混淆矩阵计算。",
          "  - **95% CI**：Wilson 区间（与主评测 FPR 同口径）。",
          "- `apply_audits.py`（回填后运行）可消费该模板，产出 agreement / κ / CI。", "",
          "> **本阶段只准备材料,不产生标签。** 与数据集'无人工真值'的诚实声明配套使用。"]
    (REPORTS / "human_audit_packet.md").write_text("\n".join(R) + "\n")
    print(f"wrote audit/ (manifest + packet + template) + reports/human_audit_packet.md", flush=True)


if __name__ == "__main__":
    main()
