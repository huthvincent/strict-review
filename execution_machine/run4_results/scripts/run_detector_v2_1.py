"""RUN4 detector_v2.1 runner. Records behavior metrics (§6): finding count, severity/source/
conf-band distribution, route-invariant count, gate_predicate. Ledger via usage_ledger.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mpb_common import BASE, SPLITS  # noqa: E402
import eval_harness as H  # noqa: E402
import detector_v2_1 as D  # noqa: E402

_lock = threading.Lock()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="v2_smoke")
    ap.add_argument("--detector-id", default=None)
    ap.add_argument("--out-name", default=None)
    ap.add_argument("--ablate", default=None)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-cost-usd", type=float, default=300.0)
    a = ap.parse_args()

    cfg = json.loads(D.CONFIG_PATH.read_text()) if D.CONFIG_PATH.exists() else D.DEFAULT_CONFIG
    ablate = a.ablate.split(",") if a.ablate else []
    det_id = a.detector_id or ("detector_v2_1" if not ablate else "v2_1_ablate_" + "_".join(sorted(ablate)))
    detect = D.make_detect(cfg, ablate)
    cards, pairs, negs = H.load_cards(), H.load_pairs(), H.load_negs()
    items = H.load_items(SPLITS / f"{a.split}.txt", cards, pairs, negs)
    out_name = a.out_name or a.split
    out = BASE / "predictions" / det_id / f"{out_name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["item_id"] for l in out.read_text().splitlines()} if out.exists() else set()
    todo = [it for it in items if it["item_id"] not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"[{det_id}] {out_name}: {len(items)} items, {len(done)} done, {len(todo)} to run, ablate={ablate}", flush=True)
    st = {"n": 0, "leak": 0, "cost": 0.0, "route_inv": 0, "gatepred": 0, "stop": False, "err": 0}

    def work(it):
        if st["stop"]:
            return None
        view = H.build_view(it["repo"], it["target_sha"])
        tools = H.Tools(it["repo"], it["target_sha"], view["parent_sha"])
        t0 = time.time()
        try:
            findings, meta = detect(view, tools)
        except Exception as e:
            findings, meta = [], {"error": str(e)[:200], "cost": 0.0, "tokens": {"in": 0, "out": 0}}
        return {"item_id": it["item_id"], "kind": it["kind"], "repo": it["repo"],
                "findings": findings, "leak_attempt": tools.leak_attempt,
                "latency_s": round(time.time() - t0, 2), **meta}

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(work, it): it for it in todo}
        for f in as_completed(futs):
            rec = f.result()
            if rec is None:
                continue
            with _lock:
                st["n"] += 1; st["cost"] += rec.get("cost", 0)
                if rec.get("leak_attempt"):
                    st["leak"] += 1
                if rec.get("route_invariant_violation"):
                    st["route_inv"] += 1
                if rec.get("gate_predicate"):
                    st["gatepred"] += 1
                if rec.get("error"):
                    st["err"] += 1
                if st["cost"] >= a.max_cost_usd:
                    st["stop"] = True
                with open(out, "a") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if st["n"] % 25 == 0:
                print(f"  {st['n']}/{len(todo)} route_inv={st['route_inv']} leak={st['leak']} err={st['err']} ~${st['cost']:.2f}", flush=True)
    print(f"[{det_id}] DONE: {st['n']} items, leak={st['leak']}, route_inv={st['route_inv']}, "
          f"gate_pred={st['gatepred']}, err={st['err']}, ~${st['cost']:.2f} → {out}", flush=True)


if __name__ == "__main__":
    main()
