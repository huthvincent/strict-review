"""Stage 3 · Leg 3 — risk routing (RUN2_INSTRUCTIONS.md §3.3).

Targets low-detectability regressions: instead of guessing a bug, emit an actionable
risk signal that precisely triggers Megatron's EXISTING perf CI. This is the front half
of the "find AND verify" loop — we don't run GPUs, we route to the right recipe.

Output: severity=important finding, claim="此改动触及 {leaf}，该类问题历史上只在
{manifest 条件} 下显现"; suggested_benchmark = recipe (gpt-perf/moe_perf/hybrid-perf/
determinism-perf) + config hint. Evidence = train same-leaf manifest_conditions.

detectors/leg3_recipe_map.json (leaf → recipe + trigger conditions) is a deliverable
for NVIDIA. Classification of the item's leaf uses the PR-time view only (a light Opus
classifier over diff/paths), never the item's own label.
"""
from __future__ import annotations

import argparse
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
MAP_PATH = BASE / "detectors" / "leg3_recipe_map.json"
_client = None

# Megatron perf recipes that actually exist in the clone (verified):
#   gpt-perf(.dp8), hybrid-perf(.ep8), determinism-perf, module_performance, moe
# Map each of the 11 taxonomy categories to the recipe most likely to surface it.
CATEGORY_RECIPE = {
    "collective-comm":        ("gpt-perf",         "multi-node TP/PP; watch all-reduce/all-gather time"),
    "concurrency-sync":       ("determinism-perf", "stream/sync ordering; determinism harness exposes stalls"),
    "host-overhead":          ("gpt-perf",         "CPU-bound launch overhead; small-batch step time"),
    "kernel-efficiency":      ("module_performance", "per-module kernel microbench"),
    "compilation":            ("gpt-perf",         "first-step/compile time; torch.compile recompilation"),
    "memory-management":      ("gpt-perf-dp8",     "allocator/fragmentation; peak-mem under DP8"),
    "memory-footprint":       ("gpt-perf-dp8",     "activation/param memory; OOM threshold"),
    "parallelism-scheduling": ("hybrid-perf",      "PP schedule/bubble; interleaved 1F1B"),
    "io-startup":             ("gpt-perf",         "dataloader/ckpt startup; time-to-first-step"),
    "inference-serving":      ("gpt-perf",         "decode throughput/latency (proxy via gpt-perf)"),
    "config-observability":   ("gpt-perf",         "config-gated regressions; run with feature toggled"),
}


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


def cmd_build_map(a):
    """Build leaf → {category, recipe, config_hint, manifest_conditions[]} from train."""
    l2c = _leaf_to_category()
    cards = {c["case_id"]: c for c in read_jsonl(CASES / "cards_final_v011.jsonl")}
    train = [l.strip() for l in (SPLITS / "train.txt").read_text().splitlines() if l.strip()]
    by_leaf_manifest = defaultdict(list)
    for l in train:
        if not l.startswith("case:"):
            continue
        c = cards.get(l.split(":", 1)[1])
        if not c or not c.get("is_perf_related"):
            continue
        leaf = c.get("taxonomy_label")
        mc = c.get("manifest_conditions")
        if leaf and isinstance(mc, dict):
            by_leaf_manifest[leaf].append(mc)

    recipe_map = {}
    for leaf, cat in l2c.items():
        recipe, hint = CATEGORY_RECIPE.get(cat, ("gpt-perf", ""))
        # summarize manifest conditions across train same-leaf cards.
        # manifest values may be str OR list — normalize to hashable strings.
        conds = by_leaf_manifest.get(leaf, [])

        def _vals(field):
            out = []
            for c in conds:
                v = c.get(field)
                if isinstance(v, list):
                    out += [str(x) for x in v if x]
                elif v:
                    out.append(str(v))
            return out
        par = Counter(_vals("parallelism"))
        dt = Counter(_vals("dtype"))
        recipe_map[leaf] = {
            "category": cat, "recipe": recipe, "config_hint": hint,
            "n_train_cases": len(conds),
            "top_parallelism": [k for k, _ in par.most_common(3)],
            "top_dtype": [k for k, _ in dt.most_common(2)],
        }
    MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    MAP_PATH.write_text(json.dumps(recipe_map, ensure_ascii=False, indent=2))
    print(f"leg3 recipe map: {len(recipe_map)} leaves → {MAP_PATH}")
    rc = Counter(v["recipe"] for v in recipe_map.values())
    print("recipe distribution:", dict(rc))


def _cl():
    global _client
    if _client is None:
        from anthropic import AnthropicBedrock
        _client = AnthropicBedrock(aws_region=os.environ.get("AWS_REGION", "us-east-1"),
                                   timeout=120.0, max_retries=4)
    return _client


CLASSIFY_TOOL = {
    "name": "classify_change",
    "description": "Classify which perf-risk taxonomy leaf (if any) this diff most plausibly touches.",
    "input_schema": {"type": "object", "properties": {
        "touches_perf_surface": {"type": "boolean"},
        "leaf": {"type": ["string", "null"], "description": "best-guess taxonomy leaf id, or null"},
        "why": {"type": "string"},
        "confidence": {"type": "number"}},
        "required": ["touches_perf_surface"]},
}


def make_detect(map_path=None, leaves=None, conf_gate=0.0):
    """conf_gate: only emit a route when the classifier's own confidence ≥ gate
    (higher gate = fewer false routes on benign perf-surface changes)."""
    rm = json.loads((map_path or MAP_PATH).read_text())
    leaf_ids = leaves or sorted(rm.keys())
    l2c = _leaf_to_category()

    sys_txt = ("你是性能风险路由器。判断这个 diff 触及了哪一类性能敏感面（taxonomy 叶），"
               "以便触发正确的性能 CI。你不必断定它一定有 bug —— 只需判断它落在哪个高风险类别。"
               "可用的叶 id: " + ", ".join(leaf_ids) +
               "。若明显与性能无关（纯文档/测试/注释），touches_perf_surface=false。"
               "只调用 classify_change 一次。")

    def _view_msg(view):
        return (f"COMMIT {view['repo']}@{view['sha'][:12]}\n=== MESSAGE ===\n{view['commit_message']}\n\n"
                f"=== CHANGED FILES ===\n" + "\n".join(view['changed_files']) + "\n\n"
                f"=== DIFF ===\n{view['diff'][:6000]}\n")

    from detectors_baseline import CACHE

    def detect(view, tools):
        with _cl().messages.stream(
                model=OPUS, max_tokens=500, thinking={"type": "disabled"},
                system=[{"type": "text", "text": sys_txt, "cache_control": CACHE}],
                messages=[{"role": "user", "content": _view_msg(view)}],
                tools=[CLASSIFY_TOOL],
                tool_choice={"type": "tool", "name": "classify_change"}) as st:
            r = st.get_final_message()
        inp = next((b.input for b in r.content if b.type == "tool_use"), None) or {}
        cost = (r.usage.input_tokens * PRICE_IN + r.usage.output_tokens * PRICE_OUT) / 1e6
        if (not inp.get("touches_perf_surface") or not inp.get("leaf")
                or inp.get("confidence", 0) < conf_gate):
            return [], {"n_turns": 1, "tokens": {"in": r.usage.input_tokens, "out": r.usage.output_tokens},
                        "cost": round(cost, 5), "routed_leaf": None}
        leaf = inp["leaf"]
        info = rm.get(leaf, rm.get(l2c.get(leaf, ""), {}))
        recipe = info.get("recipe", "gpt-perf")
        conds = info.get("top_parallelism") or []
        manifest = (", ".join(conds)) if conds else "该类的典型并行/精度配置"
        finding = {
            "severity": "important",
            "category": f"risk-route:{leaf}",
            "file": None, "line": None,
            "claim": f"此改动触及 {leaf}（{info.get('category','?')}），该类问题历史上主要在 "
                     f"[{manifest}] 配置下显现；{inp.get('why','')[:160]}",
            "confidence": min(0.75, inp.get("confidence", 0.5)),
            "suggested_benchmark": f"{recipe} — {info.get('config_hint','')}",
        }
        return [finding], {"n_turns": 1, "tokens": {"in": r.usage.input_tokens, "out": r.usage.output_tokens},
                           "cost": round(cost, 5), "routed_leaf": leaf, "recipe": recipe}
    return detect


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build-map")
    a = ap.parse_args()
    if a.cmd == "build-map":
        cmd_build_map(a)


if __name__ == "__main__":
    main()
