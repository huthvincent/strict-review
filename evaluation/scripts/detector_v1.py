"""Stage 3 · Fusion + adversarial verify — detector_v1 (RUN2_INSTRUCTIONS.md §3.4).

Pipeline per item (PR-time view only):
  1. leg1 (static rules, LLM-free) ALWAYS runs first — cheapest.
  2. light route: classify high/medium/low scenario to order leg2/leg3.
     - leg2 (retrieval review) is the main leg → runs for medium/high.
     - leg3 (risk routing) runs for low (or when leg1+leg2 found nothing).
  3. merge + dedup: same (file, mechanism) → keep highest confidence.
  4. ADVERSARIAL VERIFY: a 2nd Opus (adversarial_verify.v1) tries to refute each finding.
     refuted → drop; residual_severity downgrade (critical→important→suggestion→drop).
  5. budget trim: keep top-2 by confidence (severity∈{critical,important} first).

Config frozen to detectors/detector_v1.config.json. detect() is harness-compatible.

Usage (dev build/tune):  uv run python detector_v1.py --split dev --limit N --workers W
Ablations via --ablate leg1|leg2|leg3|adversarial (Stage 4).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mpb_common import BASE, PROMPTS, SPLITS, now_iso  # noqa: E402
import eval_harness as H  # noqa: E402
import run_rules as LEG1
import leg2_retrieval as LEG2
import leg3_routing as LEG3
from detectors_baseline import DET_TOOLS, _run_tool, CACHE  # noqa: E402

OPUS = "us.anthropic.claude-opus-4-8"
PRICE_IN, PRICE_OUT = 5.0, 25.0
CONFIG_PATH = BASE / "detectors" / "detector_v1.config.json"
_cost_lock = threading.Lock()
_client = None

DEFAULT_CONFIG = {
    "version": "detector_v1",
    "leg2_k": 5, "leg2_max_turns": 4,
    "confidence_floor": 0.45,          # drop findings below this before verify
    "budget": 2,
    "adversarial": True,
    "prompts": {"leg2": "leg2 inline", "leg3": "leg3 inline",
                "adversarial": "adversarial_verify.v1"},
    "rules": "ruleset.v1/kept.jsonl",
}

VERIFY_TOOL = {
    "name": "emit_verdict",
    "description": "Adversarial verdict on a single finding.",
    "input_schema": {"type": "object", "properties": {
        "refuted": {"type": "boolean"},
        "reason": {"type": "string"},
        "residual_severity": {"type": ["string", "null"],
                              "enum": ["critical", "important", "suggestion", None]}},
        "required": ["refuted", "reason"]},
}


def _cl():
    global _client
    if _client is None:
        from anthropic import AnthropicBedrock
        _client = AnthropicBedrock(aws_region=os.environ.get("AWS_REGION", "us-east-1"),
                                   timeout=120.0, max_retries=4)
    return _client


def _norm_mech(claim):
    return re.sub(r"[^a-z0-9]+", " ", (claim or "").lower()).strip()[:80]


def _dedup(findings):
    """Same (file, normalized-mechanism-prefix) → keep highest confidence."""
    best = {}
    for f in findings:
        key = (f.get("file"), _norm_mech(f.get("claim"))[:40])
        if key not in best or f.get("confidence", 0) > best[key].get("confidence", 0):
            best[key] = f
    return list(best.values())


SEV_ORDER = {"critical": 3, "important": 2, "suggestion": 1}
DOWNGRADE = {"critical": "important", "important": "suggestion", "suggestion": None}


def _verify_prompt():
    return (PROMPTS / "adversarial_verify.v1.md").read_text()


def _adversarial(view, finding, tools):
    """Return (kept_finding_or_None, cost). Refuted → None; residual → downgrade."""
    vp = _verify_prompt()
    u = (f"COMMIT {view['repo']}@{view['sha'][:12]} (parent {view['parent_sha'][:12]})\n"
         f"=== DIFF ===\n{view['diff'][:6000]}\n\n"
         f"=== FINDING UNDER SCRUTINY ===\n"
         f"severity={finding.get('severity')} file={finding.get('file')}\n"
         f"claim: {finding.get('claim')}\n\nTry to refute it.")
    with _cl().messages.stream(
            model=OPUS, max_tokens=700, thinking={"type": "disabled"},
            system=[{"type": "text", "text": vp, "cache_control": CACHE}],
            messages=[{"role": "user", "content": u}],
            tools=[VERIFY_TOOL],
            tool_choice={"type": "tool", "name": "emit_verdict"}) as st:
        r = st.get_final_message()
    inp = next((b.input for b in r.content if b.type == "tool_use"), None) or {}
    cost = (r.usage.input_tokens * PRICE_IN + r.usage.output_tokens * PRICE_OUT) / 1e6
    if inp.get("refuted"):
        return None, cost, inp.get("reason", "")
    rs = inp.get("residual_severity")
    if rs and SEV_ORDER.get(rs, 3) < SEV_ORDER.get(finding.get("severity"), 2):
        finding = {**finding, "severity": rs, "downgraded_by_verify": True}
    return finding, cost, inp.get("reason", "")


def _trim(findings, budget):
    sev = [f for f in findings if f.get("severity") in ("critical", "important")]
    sev.sort(key=lambda f: (SEV_ORDER.get(f.get("severity"), 0), f.get("confidence", 0)), reverse=True)
    rest = [f for f in findings if f not in sev]
    rest.sort(key=lambda f: f.get("confidence", 0), reverse=True)
    return (sev + rest)[:budget]


def make_detect(config=None, ablate=None):
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    ablate = set(ablate or [])
    leg1 = LEG1.make_detect(BASE / "rules" / cfg["rules"]) if "leg1" not in ablate else None
    _kb = (BASE / "detectors" / cfg["leg2_kb"]) if cfg.get("leg2_kb") else None
    leg2 = LEG2.make_detect(kb_path=_kb, k=cfg["leg2_k"], max_turns=cfg["leg2_max_turns"]) if "leg2" not in ablate else None
    leg3 = LEG3.make_detect(conf_gate=cfg.get("leg3_conf_gate", 0.0)) if "leg3" not in ablate else None
    do_adv = cfg["adversarial"] and "adversarial" not in ablate

    def detect(view, tools):
        meta = {"legs": [], "cost": 0.0, "tokens": {"in": 0, "out": 0}, "verify": []}
        collected = []
        # leg1 first (cheap, LLM-free)
        if leg1:
            f1, m1 = leg1(view, tools)
            for f in f1:
                f["_leg"] = "leg1"
            collected += f1; meta["legs"].append("leg1")
        # leg2 main review
        if leg2:
            f2, m2 = leg2(view, tools)
            meta["cost"] += m2.get("cost", 0)
            for f in f2:
                f["_leg"] = "leg2"
            collected += f2; meta["legs"].append("leg2")
            meta["retrieved_leaves"] = m2.get("retrieved_leaves")
        # leg3 routing — run when nothing solid yet, or always if it's the only LLM leg
        strong = [f for f in collected if f.get("confidence", 0) >= cfg["confidence_floor"]
                  and f.get("severity") in ("critical", "important")]
        if leg3 and (not strong or leg2 is None):
            f3, m3 = leg3(view, tools)
            meta["cost"] += m3.get("cost", 0)
            for f in f3:
                f["_leg"] = "leg3"
            collected += f3; meta["legs"].append("leg3")
            meta["routed_leaf"] = m3.get("routed_leaf")

        # dedup + confidence floor
        merged = _dedup(collected)
        merged = [f for f in merged if f.get("confidence", 0) >= cfg["confidence_floor"]
                  or f.get("_leg") == "leg3"]  # leg3 signals kept (they're risk routes)

        # adversarial verify — applies ONLY to leg2 (LLM review claims that can be wrong).
        # leg1 = dev-validated deterministic rules (precision≥0.5 already), and leg3 =
        # "suggest a benchmark" risk routes (not bug assertions) — both bypass the refuter.
        # This targets the refuter where false alarms actually come from without gutting
        # the high-precision legs. (dev-tuned; test untouched.)
        if do_adv and merged:
            kept = []
            for f in merged:
                if cfg.get("verify_legs", ["leg2"]) and f.get("_leg") not in cfg.get("verify_legs", ["leg2"]):
                    kept.append(f)  # leg1/leg3 pass through
                    continue
                vf, vcost, vreason = _adversarial(view, f, tools)
                meta["cost"] += vcost
                meta["verify"].append({"claim": f.get("claim", "")[:60], "refuted": vf is None,
                                       "reason": vreason[:100], "leg": f.get("_leg")})
                if vf is not None:
                    kept.append(vf)
            merged = kept

        final = _trim(merged, cfg["budget"])
        # strip internal keys
        for f in final:
            f.pop("_leg", None)
        return final, {"n_turns": meta.get("legs"), "legs": meta["legs"],
                       "cost": round(meta["cost"], 5),
                       "verify": meta["verify"], "tokens": meta["tokens"],
                       "retrieved_leaves": meta.get("retrieved_leaves"),
                       "routed_leaf": meta.get("routed_leaf")}
    return detect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-cost-usd", type=float, default=200.0)
    ap.add_argument("--ablate", default=None, help="comma list: leg1,leg2,leg3,adversarial")
    ap.add_argument("--detector-id", default=None)
    a = ap.parse_args()

    cfg = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else DEFAULT_CONFIG
    ablate = a.ablate.split(",") if a.ablate else []
    det_id = a.detector_id or ("detector_v1" if not ablate else "ablate_" + "_".join(sorted(ablate)))
    detect = make_detect(cfg, ablate)

    cards, pairs, negs = H.load_cards(), H.load_pairs(), H.load_negs()
    items = H.load_items(SPLITS / f"{a.split}.txt", cards, pairs, negs)
    out = BASE / "predictions" / det_id / f"{a.split}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["item_id"] for l in out.read_text().splitlines()} if out.exists() else set()
    todo = [it for it in items if it["item_id"] not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"[{det_id}] {a.split}: {len(items)} items, {len(done)} done, {len(todo)} to run, "
          f"ablate={ablate}", flush=True)

    state = {"spent": 0.0, "n": 0, "leaks": 0, "stop": False}
    wlock = threading.Lock()

    def work(it):
        if state["stop"]:
            return None
        import time
        view = H.build_view(it["repo"], it["target_sha"])
        toolset = H.Tools(it["repo"], it["target_sha"], view["parent_sha"])
        t0 = time.time()
        try:
            findings, meta = detect(view, toolset)
        except Exception as e:
            findings, meta = [], {"error": str(e)[:200], "cost": 0.0, "tokens": {"in": 0, "out": 0}}
        return {"item_id": it["item_id"], "kind": it["kind"], "repo": it["repo"],
                "findings": findings, "leak_attempt": toolset.leak_attempt,
                "latency_s": round(time.time() - t0, 2), **meta}

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(work, it): it for it in todo}
        for f in as_completed(futs):
            rec = f.result()
            if rec is None:
                continue
            with _cost_lock:
                state["spent"] += rec.get("cost", 0); state["n"] += 1
                if rec.get("leak_attempt"):
                    state["leaks"] += 1
                if state["spent"] >= a.max_cost_usd:
                    state["stop"] = True
            with wlock:
                with open(out, "a") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if state["n"] % 100 == 0:
                print(f"  {state['n']}/{len(todo)} ({state['leaks']} leaks) ~${state['spent']:.2f}", flush=True)
    print(f"[{det_id}] DONE: {state['n']} items, {state['leaks']} leaks, ~${state['spent']:.2f} → {out}",
          flush=True)


if __name__ == "__main__":
    main()
