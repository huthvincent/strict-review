"""RUN3 Stage 2.1 — full 74-leaf handbook (knowledge/handbook.v1.md), train-only.

Each leaf gets one page distilled BY OPUS from that leaf's train perf cards ONLY:
  典型反模式 (2-4) / 显现条件 top3 / 历史量级样本 / 检测时该查什么(父快照核验要点) / 高发文件 top5.

Anti-leak (§2.1①): the distillation prompt FORBIDS citing any concrete case/commit-sha/
issue/PR/date NOT present in the provided train cards (incl. the model's parametric-memory
"famous cases"). A separate audit (handbook_leak_audit.py) regex-scans for sha/issue/PR/date
and cross-checks against the train-card set.

Callable per-leaf: distill_leaf(leaf) -> {leaf, page_md, provenance}. The Workflow driver
fans this out across 74 leaves + does the 10-leaf second-distillation cross-read.
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mpb_common import BASE, CASES, SPLITS, read_jsonl, now_iso  # noqa: E402

OPUS = "us.anthropic.claude-opus-4-8"
PRICE_IN, PRICE_OUT = 5.0, 25.0
_client = None


def _cl():
    global _client
    if _client is None:
        from anthropic import AnthropicBedrock
        _client = AnthropicBedrock(aws_region=os.environ.get("AWS_REGION", "us-east-1"),
                                   timeout=150.0, max_retries=4)
    return _client


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


def train_cards_by_leaf():
    cards = {c["case_id"]: c for c in read_jsonl(CASES / "cards_final_v011.jsonl")}
    train = set(l.strip().split(":", 1)[1] for l in (SPLITS / "train.txt").read_text().splitlines()
                if l.strip().startswith("case:"))
    by = defaultdict(list)
    for cid, c in cards.items():
        if cid in train and c.get("is_perf_related") and c.get("taxonomy_label"):
            by[c["taxonomy_label"]].append(c)
    return by


HANDBOOK_TOOL = {
    "name": "emit_leaf_page",
    "description": "Emit the handbook page for this taxonomy leaf, distilled ONLY from the provided train cards.",
    "input_schema": {"type": "object", "properties": {
        "antipatterns": {"type": "array", "items": {"type": "string"},
                         "description": "2-4 concrete code antipatterns typical of this leaf"},
        "manifest_conditions_top3": {"type": "array", "items": {"type": "string"},
                                     "description": "top-3 conditions under which the regression manifests"},
        "magnitude_samples": {"type": "array", "items": {"type": "string"},
                              "description": "historical magnitude samples (only if present in cards)"},
        "detection_checklist": {"type": "array", "items": {"type": "string"},
                                "description": "what to check at parent-snapshot (hot path? caller? default-enabled?)"},
        "high_freq_files_top5": {"type": "array", "items": {"type": "string"},
                                 "description": "top-5 file paths/globs frequently involved"},
    }, "required": ["antipatterns", "manifest_conditions_top3", "detection_checklist"]},
}

SYSTEM = """你是性能回归检测手册的编纂者。给你**某一个 taxonomy 叶**的一组**训练集卡片**
（每张含机制/症状/显现条件/量级/高发文件线索）。你的任务：把它们蒸馏成该叶的一页手册。

**铁律（违反即作废）**：
- 只准使用**我提供的这些卡片**里的信息。**严禁**引用任何不在这些卡片里的具体案例、
  commit sha、issue/PR 号、日期——包括你从参数记忆里"记得的著名案例"。宁可写抽象反模式，
  也不要编造或从记忆里搬具体标识符。
- 反模式要**可泛化**（描述代码形态与机制），不要写死具体变量名/文件名。
- 高发文件写**路径模式/目录**（从卡片的 repo+机制推断），不是编造的完整路径。
- 若某一栏卡片信息不足，就少写或留空，不要凑。

通过 emit_leaf_page 输出。"""


def _card_brief(c):
    return (f"- [{c['repo']}] kind={c.get('kind')} detectability={c.get('static_detectability')}\n"
            f"  机制: {(c.get('mechanism') or '')[:280]}\n"
            f"  症状: {c.get('symptom')} · 量级: {c.get('magnitude_reported') or 'n/a'}\n"
            f"  显现条件: {str(c.get('manifest_conditions') or {})[:200]}")


def distill_leaf(leaf, category, cards, seed_tag="A"):
    """Distill one leaf page from its train cards. Returns dict or None if 0 cards."""
    if not cards:
        return {"leaf": leaf, "category": category, "n_cards": 0, "skipped": True,
                "reason": "no train perf cards for this leaf"}
    # cap cards fed to prompt for cost; keep a representative sample (all if <=25)
    sample = cards[:25]
    # high-freq files from ALL cards (deterministic, not LLM) — file basenames from sha diffs
    # (we pass the mechanism-derived hint to the LLM; the deterministic file list is added by profiler)
    user = (f"叶子 id: {leaf}（大类: {category}）\n"
            f"训练卡片数: {len(cards)}（下方展示前 {len(sample)} 张）\n\n"
            + "\n".join(_card_brief(c) for c in sample))
    with _cl().messages.stream(model=OPUS, max_tokens=1800, thinking={"type": "disabled"},
                               system=SYSTEM, messages=[{"role": "user", "content": user}],
                               tools=[HANDBOOK_TOOL],
                               tool_choice={"type": "tool", "name": "emit_leaf_page"}) as st:
        r = st.get_final_message()
    inp = next((b.input for b in r.content if b.type == "tool_use"), None) or {}
    cost = (r.usage.input_tokens * PRICE_IN + r.usage.output_tokens * PRICE_OUT) / 1e6
    return {"leaf": leaf, "category": category, "n_cards": len(cards), "skipped": False,
            "page": inp, "src_case_ids": [c["case_id"] for c in cards],
            "provenance": {"model": OPUS, "prompt": "handbook.v1", "seed_tag": seed_tag,
                           "ts": now_iso(), "cost": round(cost, 5),
                           "tokens": {"in": r.usage.input_tokens, "out": r.usage.output_tokens}}}


if __name__ == "__main__":
    # standalone smoke: distill 1 leaf
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--leaf", default=None)
    a = ap.parse_args()
    by = train_cards_by_leaf()
    l2c = _leaf_to_category()
    leaf = a.leaf or sorted(by, key=lambda k: -len(by[k]))[0]
    out = distill_leaf(leaf, l2c.get(leaf, "?"), by.get(leaf, []))
    print(json.dumps(out, ensure_ascii=False, indent=2)[:2000])
