"""RUN4 §1.2 liveness gate — run4.v3 scoring (post-BLOCKERS decision).
Recomputed from EXISTING smoke predictions (no re-run). Four conditions:
  ① ≥8/18 pos fire a severe finding (critical/important), NO conf threshold (metrics.v1)
  ② ≤3/7 neg severe∧conf_raw≥0.5 false positives
  ③ route invariant = 0 (touches=true & no route finding)
  ④ leak = 0
conf_final (×0.7 discount) is ONLY for budget-2 sorting; it does NOT enter scoring.
The gate's recall side is source-agnostic severe-fire (same口径 as v1's 25.3%/13.3% line).
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mpb_common import BASE, now_iso  # noqa: E402

SEV = {"critical", "important"}


def severe_findings(rec):
    """budget-2 output severe findings (no conf threshold) — recall side, metrics.v1."""
    return [f for f in rec["findings"] if f.get("severity") in SEV]


def neg_fp_findings(rec):
    """误报侧 = severe ∧ conf_raw ≥ 0.5."""
    return [f for f in rec["findings"] if f.get("severity") in SEV and f.get("conf_raw", 0) >= 0.5]


def main():
    det = "detector_v2_1"
    rows = [json.loads(l) for l in (BASE / "predictions" / det / "v2_smoke.jsonl").read_text().splitlines()]
    pos = [r for r in rows if r["kind"] in ("case", "pair")]
    neg = [r for r in rows if r["kind"] == "neg"]

    # ① pos severe fire (NO conf threshold; source-agnostic) — no judge needed for liveness gate
    pos_fire = sum(1 for r in pos if severe_findings(r))
    fire_ids = [r["item_id"] for r in pos if severe_findings(r)]
    # ② neg severe∧conf_raw≥0.5 FP
    neg_fp = sum(1 for r in neg if neg_fp_findings(r))
    neg_fp_ids = [r["item_id"] for r in neg if neg_fp_findings(r)]
    # ③ route invariant ; ④ leak
    route_inv = sum(1 for r in rows if r.get("route_invariant_violation"))
    leak = sum(1 for r in rows if r.get("leak_attempt"))

    c1 = pos_fire >= 8; c2 = neg_fp <= 3; c3 = route_inv == 0; c4 = leak == 0
    allpass = c1 and c2 and c3 and c4

    # behavior metrics (§6)
    src = Counter(f.get("source") for r in rows for f in r["findings"])
    sev = Counter(f.get("severity") for r in rows for f in r["findings"])
    band_raw = Counter(("≥0.5" if f.get("conf_raw", 0) >= 0.5 else "<0.5") for r in rows for f in r["findings"])
    band_final = Counter(("≥0.5" if f.get("conf_final", 0) >= 0.5 else "<0.5") for r in rows for f in r["findings"])
    with_finding = sum(1 for r in rows if r["findings"])
    unit = sum(r.get("cost", 0) for r in rows) / max(1, len(rows))

    L = ["# RUN4 §1.2 活性门（冒烟 18 正 + 7 负，**run4.v3 口径**）", "",
         f"- generated: {now_iso()} · **已存冒烟预测直接重算，未重跑**（§1.2）",
         f"- 口径：召回侧 = severe 开火无 conf 门槛（metrics.v1）；误报侧 = severe∧conf_raw≥0.5；conf_final 只用于排序",
         "", "## 冻结门四条（run4.v3）", "",
         f"- ① 正样本 severe 开火（无门槛）**{pos_fire}/18** (需≥8): {'✓' if c1 else '✗'}",
         f"- ② 负样本 severe∧conf_raw≥0.5 误报 **{neg_fp}/7** (需≤3): {'✓' if c2 else '✗'}"
         + (f"  误报项: {neg_fp_ids}" if neg_fp_ids else ""),
         f"- ③ route 不变量违规 **{route_inv}** (需=0): {'✓' if c3 else '✗'}",
         f"- ④ leak **{leak}** (需=0): {'✓' if c4 else '✗'}",
         f"- **四条全过: {'✓ 过门 → FROZEN' if allpass else '✗ 不过'}**"
         + ("" if allpass or c2 else "（②在新口径下 >3/7 → 仍 BLOCKERS 回报，§1.2 行119）"),
         "", "## 行为指标（§6 必披露）", "",
         f"- 出 finding 项数: {with_finding}/{len(rows)} · $/item ~${unit:.3f}（<$0.85 硬触发线）",
         f"- source 分布: {dict(src)}",
         f"- severity 分布: {dict(sev)}",
         f"- conf_raw 带分布: {dict(band_raw)} · conf_final 带分布: {dict(band_final)}",
         f"- 正样本 severe 开火项: {fire_ids}"]
    (BASE / "reports" / "liveness_gate.md").write_text("\n".join(L) + "\n")
    (BASE / "reports" / "liveness_gate.json").write_text(json.dumps(
        {"scoring": "run4.v3", "pos_fire": pos_fire, "neg_fp": neg_fp, "route_inv": route_inv,
         "leak": leak, "pass": allpass, "unit_cost": round(unit, 4)}, indent=2))
    print("\n".join(L))
    print(f"\n>>> LIVENESS GATE (run4.v3): {'PASS' if allpass else 'FAIL'}")


if __name__ == "__main__":
    main()
