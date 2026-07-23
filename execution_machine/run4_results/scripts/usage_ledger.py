"""RUN4 token/cost ledger (§ header + §1.1). Every LLM call appends REAL in/out tokens +
cost to reports/usage_ledger.jsonl, attributed to a Stage. The cost-reconciliation table
in run4_report MUST balance against this ledger (RUN3's $34 discrepancy must not recur).
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from mpb_common import BASE  # noqa: E402

LEDGER = BASE / "reports" / "usage_ledger.jsonl"
PRICE_IN, PRICE_OUT = 5.0, 25.0            # Opus 4.8 per-MTok (input, output)
PRICE_CW, PRICE_CR = 6.25, 0.5             # cache write (1.25x), cache read (0.1x)
_lock = threading.Lock()


def record(stage: str, label: str, usage, model="us.anthropic.claude-opus-4-8", extra=None):
    """usage = anthropic usage obj (has input_tokens/output_tokens[/cache_*]) or a dict.
    Returns the computed cost (USD). Thread-safe append."""
    def g(k, default=0):
        if usage is None:
            return default
        if isinstance(usage, dict):
            return usage.get(k, default) or default
        return getattr(usage, k, default) or default
    itok = g("input_tokens"); otok = g("output_tokens")
    cw = g("cache_creation_input_tokens"); cr = g("cache_read_input_tokens")
    cost = (itok * PRICE_IN + otok * PRICE_OUT + cw * PRICE_CW + cr * PRICE_CR) / 1e6
    rec = {"stage": stage, "label": label, "model": model,
           "in": itok, "out": otok, "cache_w": cw, "cache_r": cr, "cost": round(cost, 6)}
    if extra:
        rec.update(extra)
    with _lock:
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with open(LEDGER, "a") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return cost


def totals_by_stage():
    if not LEDGER.exists():
        return {}
    agg = {}
    for l in LEDGER.read_text().splitlines():
        try:
            r = json.loads(l)
        except Exception:
            continue
        s = r.get("stage", "?")
        a = agg.setdefault(s, {"calls": 0, "in": 0, "out": 0, "cost": 0.0})
        a["calls"] += 1; a["in"] += r.get("in", 0); a["out"] += r.get("out", 0); a["cost"] += r.get("cost", 0)
    for a in agg.values():
        a["cost"] = round(a["cost"], 4)
    return agg


if __name__ == "__main__":
    import sys
    print(json.dumps(totals_by_stage(), indent=2))
    tot = sum(a["cost"] for a in totals_by_stage().values())
    print(f"GRAND TOTAL: ${tot:.2f}")
