"""RUN3 §2.1③ (corrected) — semantic 10-leaf cross-read.

The first cross-read used a LEXICAL token-overlap metric, which flagged 10/10 REWRITE
because LLM free-text at temperature reproduces CONCEPTS, not token sets. §2.1③ asks
whether the requirement (bullet) SET differs by more than half — a SEMANTIC question.
This re-does it with an Opus judge comparing the two independent distillations
concept-by-concept, plus the required "external-content check" column.

Rewrite flag: judge says >50% of A's core points have NO semantic match in B.
"""
from __future__ import annotations

import json
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mpb_common import BASE, now_iso  # noqa: E402
import build_handbook as HB  # noqa: E402

KDIR = BASE / "knowledge"
SEED = 20260720
N = 10
_client = None


def _cl():
    global _client
    if _client is None:
        from anthropic import AnthropicBedrock
        _client = AnthropicBedrock(aws_region=os.environ.get("AWS_REGION", "us-east-1"), timeout=120, max_retries=4)
    return _client


CMP_TOOL = {
    "name": "emit_comparison",
    "description": "Compare two independent distillations of the same taxonomy leaf.",
    "input_schema": {"type": "object", "properties": {
        "concept_overlap_frac": {"type": "number",
                                 "description": "fraction of distillation A's core points that have a SEMANTIC match in B (0-1)"},
        "materially_different": {"type": "boolean",
                                 "description": "true if >50% of A's core points lack a semantic match in B (=> rewrite)"},
        "external_content_in_either": {"type": "boolean",
                                       "description": "true if EITHER distillation cites a concrete case/sha/issue/PR/date NOT derivable from generic mechanism knowledge"},
        "notes": {"type": "string"},
    }, "required": ["concept_overlap_frac", "materially_different", "external_content_in_either"]},
}

SYS = """你比较**同一个 taxonomy 叶**的两次独立蒸馏（A/B），判断它们的要点集合是否语义一致。
按语义（机制/反模式是否说的是同一回事）比较，不按措辞。输出：
- concept_overlap_frac：A 的核心要点中有多少比例能在 B 找到语义对应（0–1）。
- materially_different：是否 >50% 的 A 要点在 B 找不到语义对应（→ 需重写）。
- external_content_in_either：A 或 B 是否出现了**无法由通用机制知识推出**的具体案例/sha/issue/PR/日期
  （泄漏信号）。只调用 emit_comparison 一次。"""


def compare(leaf, a, b):
    def fmt(p):
        return (f"反模式: {p.get('antipatterns', [])}\n"
                f"检查清单: {p.get('detection_checklist', [])}\n"
                f"显现条件: {p.get('manifest_conditions_top3', [])}")
    u = f"叶: {leaf}\n\n=== 蒸馏 A ===\n{fmt(a)}\n\n=== 蒸馏 B ===\n{fmt(b)}"
    with _cl().messages.stream(model=HB.OPUS, max_tokens=400, thinking={"type": "disabled"},
                               system=SYS, messages=[{"role": "user", "content": u}],
                               tools=[CMP_TOOL], tool_choice={"type": "tool", "name": "emit_comparison"}) as st:
        r = st.get_final_message()
    return next((x.input for x in r.content if x.type == "tool_use"), {}) or {}


def main():
    pages = {json.loads(l)["leaf"]: json.loads(l) for l in (KDIR / "handbook.v1.jsonl").read_text().splitlines()}
    l2c = HB._leaf_to_category()
    by = HB.train_cards_by_leaf()
    non_skip = [l for l, p in pages.items() if not p.get("skipped")]
    rnd = random.Random(SEED)
    leaves = rnd.sample(non_skip, min(N, len(non_skip)))

    # re-distill B for each + semantic compare
    lock = threading.Lock()
    rows = []

    def work(leaf):
        b = HB.distill_leaf(leaf, l2c.get(leaf, "?"), by.get(leaf, []), "B")
        cmp = compare(leaf, pages[leaf]["page"], b.get("page", {}))
        return {"leaf": leaf, "overlap": round(cmp.get("concept_overlap_frac", 0), 2),
                "materially_different": bool(cmp.get("materially_different")),
                "external_content": bool(cmp.get("external_content_in_either")),
                "notes": cmp.get("notes", "")[:120]}

    with ThreadPoolExecutor(max_workers=6) as ex:
        for f in as_completed([ex.submit(work, l) for l in leaves]):
            rows.append(f.result())

    nrw = sum(1 for r in rows if r["materially_different"])
    nleak = sum(1 for r in rows if r["external_content"])
    R = ["# 手册 10 叶对读（语义版，§2.1③修正）", "",
         f"- generated: {now_iso()} · seed {SEED} · **语义比较**（LLM judge），非词面 token 重叠",
         f"- 前次用词面 token 重叠 → 10/10 误报 REWRITE（LLM 自由文本复现概念而非 token）。此为修正版。",
         "- 含『外源内容检查』栏（§2.1③要求）：两次蒸馏是否出现无法由通用机制推出的具体标识符。", "",
         "| 叶 | 概念重叠 | 需重写(>50%无对应)? | 外源内容? | 备注 |", "|---|--:|:--:|:--:|---|"]
    for r in sorted(rows, key=lambda r: r["overlap"]):
        R.append(f"| {r['leaf']} | {r['overlap']} | {'⚠️ REWRITE' if r['materially_different'] else '✓'} | "
                 f"{'⚠️ 有' if r['external_content'] else '✓ 无'} | {r['notes']} |")
    R += ["", f"- 需重写: **{nrw}/{len(rows)}** · 外源内容: **{nleak}/{len(rows)}**",
          ("- ✓ 手册稳定且无泄漏信号，采纳。" if nrw == 0 and nleak == 0
           else f"- 对 {nrw} 个需重写叶重蒸取交集；{nleak} 个外源信号交泄漏审计处理。")]
    (BASE / "reports" / "handbook_crossread.md").write_text("\n".join(R) + "\n")
    print("\n".join(R))


if __name__ == "__main__":
    main()
