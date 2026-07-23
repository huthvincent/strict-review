"""RUN3 detector_v2 runner (harness-compatible, concurrent). Writes
predictions/{detector_id}/{split}.jsonl. Supports --ablate handbook|tools and
--views-file for inducing-side views (Stage 4 converted double-run).
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
import detector_v2 as D2  # noqa: E402

_lock = threading.Lock()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="v2_smoke")
    ap.add_argument("--views-file", default=None, help="jsonl with prebuilt {case_id,repo,view} (inducing side)")
    ap.add_argument("--detector-id", default=None)
    ap.add_argument("--out-name", default=None, help="output jsonl name (default = split)")
    ap.add_argument("--ablate", default=None)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-cost-usd", type=float, default=250.0)
    a = ap.parse_args()

    cfg = json.loads((D2.CONFIG_PATH).read_text()) if D2.CONFIG_PATH.exists() else D2.DEFAULT_CONFIG
    ablate = a.ablate.split(",") if a.ablate else []
    det_id = a.detector_id or ("detector_v2" if not ablate else "v2_ablate_" + "_".join(sorted(ablate)))
    detect = D2.make_detect(cfg, ablate)

    if a.views_file:
        # inducing-side prebuilt views
        rows = [json.loads(l) for l in (BASE / a.views_file).read_text().splitlines()]
        items = [{"item_id": f"caseind:{r['case_id']}", "kind": "case_inducing",
                  "repo": r["repo"], "target_sha": r["inducing_sha"], "_view": r["view"]} for r in rows]
    else:
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

    st = {"n": 0, "leak": 0, "gate": 0, "cost": 0.0, "stop": False}

    def work(it):
        if st["stop"]:
            return None
        view = it.get("_view") or H.build_view(it["repo"], it["target_sha"])
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
                if rec.get("gate"):
                    st["gate"] += 1
                if st["cost"] >= a.max_cost_usd:
                    st["stop"] = True
                with open(out, "a") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if st["n"] % 50 == 0:
                print(f"  {st['n']}/{len(todo)} (leak {st['leak']}, gate {st['gate']}) ~${st['cost']:.2f}", flush=True)
    print(f"[{det_id}] DONE: {st['n']} items, {st['leak']} leaks, {st['gate']} gated, ~${st['cost']:.2f} → {out}", flush=True)


if __name__ == "__main__":
    main()
