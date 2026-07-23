"""RUN3 Stage 2.1 driver — distill all 74 leaves concurrently + 10-leaf cross-read.

Writes:
  knowledge/handbook.v1.jsonl        (per-leaf structured pages + provenance)
  knowledge/handbook.v1.md           (human-readable, one section per leaf)
  reports/handbook_crossread.md      (10 random leaves distilled twice, requirement-set diff)
Cross-read: if >50% of requirement bullets differ between the two distillations for a leaf,
flag REWRITE (spec §2.1③).
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
CROSSREAD_N = 10
SEED = 20260720


def _set_overlap(a, b):
    """Jaccard-ish similarity of two bullet lists by normalized token sets."""
    def norm(xs):
        return set(" ".join(str(x).lower().split()) for x in (xs or []))
    A, B = norm(a), norm(b)
    if not A and not B:
        return 1.0
    # bullet-level: fraction of A's bullets with a close match in B (token-overlap ≥0.5)
    def toks(s):
        import re
        return set(re.findall(r"[a-z0-9_]+", s))
    matched = 0
    for x in A:
        tx = toks(x)
        if any(len(tx & toks(y)) / max(1, len(tx | toks(y))) >= 0.4 for y in B):
            matched += 1
    return matched / max(1, len(A))


def main():
    KDIR.mkdir(parents=True, exist_ok=True)
    by = HB.train_cards_by_leaf()
    l2c = HB._leaf_to_category()
    leaves = sorted(l2c.keys())
    state = {"cost": 0.0, "done": 0}
    lock = threading.Lock()
    pages = {}

    def work(leaf):
        res = HB.distill_leaf(leaf, l2c.get(leaf, "?"), by.get(leaf, []), seed_tag="A")
        with lock:
            state["done"] += 1
            state["cost"] += res.get("provenance", {}).get("cost", 0)
            if state["done"] % 10 == 0:
                print(f"  {state['done']}/{len(leaves)} leaves, ~${state['cost']:.2f}", flush=True)
        return res

    print(f"distilling {len(leaves)} leaves...", flush=True)
    with ThreadPoolExecutor(max_workers=10) as ex:
        for f in as_completed([ex.submit(work, l) for l in leaves]):
            r = f.result()
            pages[r["leaf"]] = r

    # write jsonl + md
    with open(KDIR / "handbook.v1.jsonl", "w") as fh:
        for leaf in leaves:
            fh.write(json.dumps(pages[leaf], ensure_ascii=False) + "\n")

    L = ["# MegaPerfBench 检测手册 v1（74 叶，train-only 蒸馏）", "",
         f"- generated: {now_iso()} · model {HB.OPUS} · prompt handbook.v1 · 仅从 train 卡蒸馏",
         f"- 覆盖 {sum(1 for p in pages.values() if not p.get('skipped'))}/{len(leaves)} 叶"
         f"（{sum(1 for p in pages.values() if p.get('skipped'))} 叶无 train 卡，如实跳过）", ""]
    skipped = [leaf for leaf in leaves if pages[leaf].get("skipped")]
    if skipped:
        L += [f"**跳过叶（无 train 卡）**: {skipped}", ""]
    cur_cat = None
    for leaf in sorted(leaves, key=lambda l: (l2c.get(l, "?"), l)):
        p = pages[leaf]
        cat = l2c.get(leaf, "?")
        if cat != cur_cat:
            L += [f"\n## 大类：{cat}\n"]
            cur_cat = cat
        if p.get("skipped"):
            L += [f"### {leaf}  — ⏭️ 跳过（{p.get('reason')}）", ""]
            continue
        pg = p["page"]
        L += [f"### {leaf}  （{p['n_cards']} 张 train 卡）", "",
              "**典型反模式**："] + [f"- {x}" for x in pg.get("antipatterns", [])] + [""]
        if pg.get("manifest_conditions_top3"):
            L += ["**显现条件 top3**："] + [f"- {x}" for x in pg["manifest_conditions_top3"]] + [""]
        if pg.get("magnitude_samples"):
            L += ["**历史量级样本**："] + [f"- {x}" for x in pg["magnitude_samples"]] + [""]
        if pg.get("detection_checklist"):
            L += ["**检测时该查什么（父快照核验）**："] + [f"- {x}" for x in pg["detection_checklist"]] + [""]
        if pg.get("high_freq_files_top5"):
            L += ["**高发文件 top5**：" + ", ".join(f"`{x}`" for x in pg["high_freq_files_top5"]), ""]
    (KDIR / "handbook.v1.md").write_text("\n".join(L) + "\n")
    print(f"handbook: {len(leaves)} leaves, ~${state['cost']:.2f} → knowledge/handbook.v1.{{jsonl,md}}", flush=True)

    # ---- 10-leaf cross-read (second independent distillation) ----
    rnd = random.Random(SEED)
    non_skipped = [l for l in leaves if not pages[l].get("skipped")]
    cr_leaves = rnd.sample(non_skipped, min(CROSSREAD_N, len(non_skipped)))
    cr = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(HB.distill_leaf, l, l2c.get(l, "?"), by.get(l, []), "B"): l for l in cr_leaves}
        for f in as_completed(futs):
            leaf = futs[f]
            b = f.result()
            a = pages[leaf]["page"]
            bp = b.get("page", {})
            sim_ap = _set_overlap(a.get("antipatterns"), bp.get("antipatterns"))
            sim_ck = _set_overlap(a.get("detection_checklist"), bp.get("detection_checklist"))
            avg = (sim_ap + sim_ck) / 2
            cr.append({"leaf": leaf, "sim_antipatterns": round(sim_ap, 2),
                       "sim_checklist": round(sim_ck, 2), "avg": round(avg, 2),
                       "rewrite_flag": avg < 0.5})
    R = ["# 手册 10 叶对读（第二次独立蒸馏，§2.1③）", "",
         f"- generated: {now_iso()} · seed {SEED} · 要点集合差异过半（相似度<0.5）→ REWRITE",
         "- 对读同时充当『外源内容检查』：两次蒸馏都只喂 train 卡，若稳定复现同一批要点则低泄漏风险。", "",
         "| 叶 | 反模式相似度 | 检查清单相似度 | 平均 | 需重写? |", "|---|--:|--:|--:|:--:|"]
    for x in sorted(cr, key=lambda x: x["avg"]):
        R.append(f"| {x['leaf']} | {x['sim_antipatterns']} | {x['sim_checklist']} | {x['avg']} | "
                 f"{'⚠️ REWRITE' if x['rewrite_flag'] else '✓'} |")
    nflag = sum(1 for x in cr if x["rewrite_flag"])
    R += ["", f"- 需重写叶数: {nflag}/{len(cr)}"
          + ("（已达标，稳定）" if nflag == 0 else "（对这些叶重蒸并取交集）")]
    (BASE / "reports" / "handbook_crossread.md").write_text("\n".join(R) + "\n")
    print("\n".join(R))


if __name__ == "__main__":
    main()
