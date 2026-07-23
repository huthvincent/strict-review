"""Stage 2 — four baselines (RUN2_INSTRUCTIONS.md §2).

All share the harness detect(view, tools) interface and emit the unified Finding
schema via forced tool_choice. Runnable via detectors_baseline.py <which> <split>.

(a) baseline_megatron  — Opus 4.8, prompts/baseline_megatron.v1.md (the adversary)
(b) baseline_generic   — Opus 4.8, prompts/baseline_generic.v1.md
(c) baseline_keyword   — no LLM; message+filename keyword hit → one 'important'
(d) baseline_xfamily   — non-Anthropic (Nova/Llama) via Converse, generic prompt
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
from mpb_common import BASE, PROMPTS, SPLITS  # noqa: E402
import eval_harness as H  # noqa: E402

OPUS = "us.anthropic.claude-opus-4-8"
# (input, output, cache_write=1.25x in, cache_read=0.1x in) per MTok
PRICE = {"opus": (5.0, 25.0, 6.25, 0.5)}
MAX_TURNS = 6
CACHE = {"type": "ephemeral"}
_cost_lock = threading.Lock()
_client = None

FINDINGS_TOOL = {
    "name": "emit_findings",
    "description": "Emit all performance findings for this commit (empty list if none).",
    "input_schema": {"type": "object", "properties": {
        "findings": {"type": "array", "items": {"type": "object", "properties": {
            "severity": {"type": "string", "enum": ["critical", "important", "suggestion"]},
            "category": {"type": "string"},
            "file": {"type": ["string", "null"]},
            "line": {"type": ["integer", "null"]},
            "claim": {"type": "string"},
            "confidence": {"type": "number"},
            "suggested_benchmark": {"type": ["string", "null"]}},
            "required": ["severity", "category", "claim", "confidence"]}}},
        "required": ["findings"]},
}

# tool schemas for the detector's parent-anchored tools (agentic baselines)
DET_TOOLS = [
    {"name": "read_file_at_parent", "description": "Read a file at the pre-change (parent) commit.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    {"name": "grep_at_parent", "description": "Grep the parent snapshot.",
     "input_schema": {"type": "object", "properties": {"pattern": {"type": "string"}, "glob": {"type": "string"}}, "required": ["pattern"]}},
    {"name": "git_log_before", "description": "git log for a path strictly before this commit.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "n": {"type": "integer"}}, "required": ["path"]}},
]


def _cl():
    global _client
    if _client is None:
        from anthropic import AnthropicBedrock
        _client = AnthropicBedrock(aws_region=os.environ.get("AWS_REGION", "us-east-1"),
                                   timeout=150.0, max_retries=4)
    return _client


def _view_msg(view):
    return (f"COMMIT {view['repo']}@{view['sha'][:12]} (parent {view['parent_sha'][:12]})\n"
            f"date: {view['author_date']}\n"
            f"=== MESSAGE ===\n{view['commit_message']}\n\n"
            f"=== CHANGED FILES ===\n" + "\n".join(view['changed_files']) + "\n\n"
            f"=== DIFF ===\n{view['diff']}\n")


def _run_tool(tools: H.Tools, name, inp):
    if name == "read_file_at_parent":
        return tools.read_file_at_parent(inp.get("path", ""))
    if name == "grep_at_parent":
        return tools.grep_at_parent(inp.get("pattern", ""), inp.get("glob", ""))
    if name == "git_log_before":
        return tools.git_log_before(inp.get("path", ""), inp.get("n", 10))
    return "[unknown tool]"


def _mark_last_cache(messages):
    """Put an ephemeral cache breakpoint at the end of the current prefix so the next
    turn re-reads (not re-bills) the whole conversation. content must be block-form."""
    # strip any prior breakpoints (max 4 allowed; keep only system+tools+latest)
    for m in messages:
        if isinstance(m["content"], list):
            for b in m["content"]:
                if isinstance(b, dict):
                    b.pop("cache_control", None)
    last = messages[-1]
    if isinstance(last["content"], str):
        last["content"] = [{"type": "text", "text": last["content"], "cache_control": CACHE}]
    elif isinstance(last["content"], list) and last["content"]:
        last["content"][-1]["cache_control"] = CACHE


def _opus_detect(system_prompt):
    """Return a detect(view, tools) closure using Opus + given system prompt.

    Prompt caching: the system prompt + tool schemas are identical across all items,
    and each turn's growing prefix repeats — so we cache-mark the system block and the
    tail of the conversation. Cache reads bill at 0.1x, cutting the 6-turn re-send cost.
    """
    sys_blocks = [{"type": "text", "text": system_prompt, "cache_control": CACHE}]
    tools = DET_TOOLS + [FINDINGS_TOOL]

    def detect(view, dtools):
        messages = [{"role": "user", "content": _view_msg(view)}]
        _mark_last_cache(messages)
        itok = otok = ctok_w = ctok_r = 0
        findings = None
        turns = 0
        for turn in range(MAX_TURNS):
            turns = turn + 1
            force = turn >= MAX_TURNS - 1
            kw = dict(model=OPUS, max_tokens=2500, thinking={"type": "disabled"},
                      system=sys_blocks, messages=messages, tools=tools)
            if force:
                kw["tool_choice"] = {"type": "tool", "name": "emit_findings"}
            with _cl().messages.stream(**kw) as st:
                resp = st.get_final_message()
            u = resp.usage
            itok += u.input_tokens; otok += u.output_tokens
            ctok_w += getattr(u, "cache_creation_input_tokens", 0) or 0
            ctok_r += getattr(u, "cache_read_input_tokens", 0) or 0
            tus = [b for b in resp.content if b.type == "tool_use"]
            emit = next((b for b in tus if b.name == "emit_findings"), None)
            if emit is not None:
                findings = emit.input.get("findings", [])
                break
            if not tus:
                messages.append({"role": "assistant", "content": resp.content})
                messages.append({"role": "user", "content": "Call emit_findings now."})
                _mark_last_cache(messages)
                continue
            messages.append({"role": "assistant", "content": resp.content})
            results = [{"type": "tool_result", "tool_use_id": b.id,
                        "content": _run_tool(dtools, b.name, b.input)} for b in tus]
            messages.append({"role": "user", "content": results})
            _mark_last_cache(messages)
        pin, pout, pcw, pcr = PRICE["opus"]
        cost = (itok * pin + otok * pout + ctok_w * pcw + ctok_r * pcr) / 1e6
        return (findings or []), {"n_turns": turns,
                                  "tokens": {"in": itok, "out": otok,
                                             "cache_w": ctok_w, "cache_r": ctok_r},
                                  "cost": round(cost, 5)}
    return detect


# (c) keyword — no LLM
KW = re.compile(r'\b(perf|slow|regress|throughput|latency|memory leak|oom|overhead|optimiz)', re.I)


def detect_keyword(view, tools):
    hay = view["commit_message"] + " " + " ".join(view["changed_files"])
    m = KW.search(hay)
    if m:
        return [{"severity": "important", "category": "keyword-match",
                 "file": None, "line": None,
                 "claim": f"commit message/filenames contain performance keyword '{m.group(0)}'",
                 "confidence": 0.6, "suggested_benchmark": None}], {"n_turns": 0, "tokens": {"in": 0, "out": 0}, "cost": 0.0}
    return [], {"n_turns": 0, "tokens": {"in": 0, "out": 0}, "cost": 0.0}


# (d) cross-family via Converse — non-agentic single call (Nova/Llama don't share tool API)
def _xfamily_detect(model_id, system_prompt):
    import boto3
    def detect(view, tools):
        bc = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"))
        sys_txt = system_prompt + ('\n\nYou have NO tools here. Respond ONLY with a JSON object: '
                                   '{"findings":[{"severity":"critical|important|suggestion",'
                                   '"category":str,"file":str|null,"line":int|null,"claim":str,'
                                   '"confidence":float}]}. Empty list if no perf problem.')
        r = bc.converse(modelId=model_id, system=[{"text": sys_txt}],
                        messages=[{"role": "user", "content": [{"text": _view_msg(view)}]}],
                        inferenceConfig={"maxTokens": 2000, "temperature": 0})
        txt = r["output"]["message"]["content"][0]["text"]
        u = r.get("usage", {})
        m = re.search(r'\{.*\}', txt, re.S)
        try:
            findings = json.loads(m.group(0)).get("findings", []) if m else []
        except Exception:
            findings = []
        return findings, {"n_turns": 1, "tokens": {"in": u.get("inputTokens", 0),
                          "out": u.get("outputTokens", 0)}, "cost": 0.0, "xfamily_model": model_id}
    return detect


def get_detector(which):
    if which == "baseline_megatron":
        return _opus_detect((PROMPTS / "baseline_megatron.v1.md").read_text())
    if which == "baseline_generic":
        return _opus_detect((PROMPTS / "baseline_generic.v1.md").read_text())
    if which == "baseline_keyword":
        return detect_keyword
    if which.startswith("baseline_xfamily"):
        model = os.environ.get("MPB_XFAMILY_MODEL", "us.amazon.nova-pro-v1:0")
        return _xfamily_detect(model, (PROMPTS / "baseline_generic.v1.md").read_text())
    raise SystemExit(f"unknown baseline {which}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which")
    ap.add_argument("--split", default="test_eval_subset")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-cost-usd", type=float, default=200.0)
    a = ap.parse_args()

    cards, pairs, negs = H.load_cards(), H.load_pairs(), H.load_negs()
    items = H.load_items(SPLITS / f"{a.split}.txt", cards, pairs, negs)
    detect = get_detector(a.which)

    # concurrent driver (harness runs sequentially; wrap for throughput)
    out = BASE / "predictions" / a.which / f"{a.split}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["item_id"] for l in out.read_text().splitlines()} if out.exists() else set()
    todo = [it for it in items if it["item_id"] not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"[{a.which}] {a.split}: {len(items)} items, {len(done)} done, {len(todo)} to run, "
          f"{a.workers} workers", flush=True)
    state = {"spent": 0.0, "n": 0, "leaks": 0, "stop": False}
    wlock = threading.Lock()

    def work(it):
        if state["stop"]:
            return None
        view = H.build_view(it["repo"], it["target_sha"])
        toolset = H.Tools(it["repo"], it["target_sha"], view["parent_sha"])
        import time
        t0 = time.time()
        try:
            findings, meta = detect(view, toolset)
        except Exception as e:
            findings, meta = [], {"error": str(e)[:200], "tokens": {"in": 0, "out": 0}, "cost": 0.0}
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
            if state["n"] % 200 == 0:
                print(f"  {state['n']}/{len(todo)} ({state['leaks']} leaks), ~${state['spent']:.2f}", flush=True)
    print(f"[{a.which}] DONE: {state['n']} items, {state['leaks']} leak_attempts, "
          f"~${state['spent']:.2f} -> {out}", flush=True)


if __name__ == "__main__":
    main()
