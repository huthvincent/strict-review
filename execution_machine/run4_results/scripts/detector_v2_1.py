"""RUN4 Stage 1 — detector_v2.1, implemented per CONTRACT A–E (charter §1.1, 宪章 7.7).

Fixes the four RUN3 v2.0 defects:
  - leg3/route was decorative → CONTRACT B: route is an INDEPENDENT leg; touches_perf_surface
    ⇒ MUST emit a route finding (invariant: touches=true & no route finding == 0).
  - three-questions was a severity hard-gate → CONTRACT C: calibrator only. Model reports
    conf_raw + unverified; harness applies conf_final = conf_raw × 0.7 iff unverified non-empty.
    Severity is NEVER lowered / finding NEVER dropped for incomplete verification.
  - resident-handbook whitelisting → CONTRACT D: handbook on-demand via handbook_lookup(leaf).
  - large-commit 2-turn truncation → CONTRACT E: cheap scan ALL blocks (batched if long) →
    top-2 deep at 5 turns; record truncation/skipped blocks in meta.

Profile gate: NO code-level short-circuit. gate_predicate computed into meta; ALL items go to
main review (step 0 executed BY THE MODEL). "含门口径" is a mechanical stat over meta.

Scoring contract (charter 统一计分口径): formal finding = severity∈{critical,important} AND
conf_final≥0.5. budget=2 sorted by conf_final. Both conf_raw and conf_final stored.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mpb_common import BASE, now_iso  # noqa: E402
import eval_harness as H  # noqa: E402
from detectors_baseline import _run_tool, CACHE  # noqa: E402  (reuse parent-snapshot tool runner)
import usage_ledger as UL  # noqa: E402

OPUS = "us.anthropic.claude-opus-4-8"
CONFIG_PATH = BASE / "detector" / "detector_v2_1.config.json"
KDIR = BASE / "knowledge"
_client = None
_STAGE = os.environ.get("RUN4_STAGE", "S1")

# CONTRACT A: verbatim system prompt (§1.1). The 4 mechanical substitutions are applied:
#  ① git show/grep -> read_file_at_parent/grep_at_parent (already the tool names below)
#  ② "读 views3.json" -> harness passes the view (no such line kept)
#  ③ removed "confidence×0.7 并" from step 3 (discount done by harness)
#  ④ 74-leaf list NOT in prompt body -> provided by handbook_lookup enum
SYSTEM_V2_1 = """你是 AI infra 性能守门员 v2.1。判断该 commit 是否可能引入性能回归
（变慢/更费显存/吞吐下降/延迟上升/卡死）。

【装备（train 数据蒸馏，按需使用，不必通读）】
- 病例手册（74 类）：用 handbook_lookup(<leaf>) 取相关叶小节。
  **手册没写的模式不代表良性。**
- 事故高发地图：视图已附本仓 top 热点文件。
- 叶子验证表：给 finding 配 suggested_benchmark 用。
【父快照核验】read_file_at_parent / grep_at_parent（只允许改动前状态）。
【工作流程（按序执行）】
0. 若改动文件**全部**在 tests/docs/examples/.github 下且不碰任何
   config/recipe/defaults 文件 → 直接 no_issue，结束。
1. **分诊（必做，独立于深审）**：判断改动是否触及性能面（计算/通信/显存/
   调度/服务热路径）。**只要触及性能面，必须立一条 route finding**：
   severity=important、source="route"、category=最相关叶子（拿不准 → 填最接近的
   叶或 "uncategorized"，confidence 取下限 0.3）、claim="该改动触及〈类别〉类
   风险（一句话说明为何相关），建议验证"、confidence=分诊把握度（0.3-0.8）、
   suggested_benchmark 按验证表填。**这条 finding 不许因深审"没确认"而删除。**
   确实不触及性能面 → 无 route finding。
2. **深审**：diff >400 行或 >10 文件 → 先快扫全部块挑最可疑的 ≤2 块做父快照
   深查；小改动直接深查。深审发现的具体机制问题另立 finding（source="deep-review"）。
3. **三问校准（不是资格闸）**：对每条 finding 尽力回答三问（热路径？谁调用？
   默认配置走到吗）。答不全 → 在 unverified 字段写明哪问未核；
   **severity 只按"若为真影响多大"定级，不许因核验不全而降级或弃报**。
4. 输出：route 与 deep-review 一起按置信度排序取前 2。
   真不触及性能面 → no_issue=true。"""

# --- tool schemas ---
def _handbook_leaf_enum():
    leaves = []
    for l in (KDIR / "handbook.v1.jsonl").read_text().splitlines():
        leaves.append(json.loads(l)["leaf"])
    return sorted(leaves)


HANDBOOK_TOOL = {
    "name": "handbook_lookup",
    "description": "取某个 taxonomy 叶的手册页（反模式/检查清单/显现条件）。无效叶名会返回提示+有效叶名列表。",
    "input_schema": {"type": "object", "properties": {
        "leaf": {"type": "string", "enum": _handbook_leaf_enum()}}, "required": ["leaf"]},
}
DET_TOOLS = [
    {"name": "read_file_at_parent", "description": "读父快照（改动前）某文件内容。",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    {"name": "grep_at_parent", "description": "在父快照 grep。",
     "input_schema": {"type": "object", "properties": {"pattern": {"type": "string"}, "glob": {"type": "string"}}, "required": ["pattern"]}},
    {"name": "git_log_before", "description": "某路径在本 commit 之前的 git log。",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "n": {"type": "integer"}}, "required": ["path"]}},
]
EMIT_TOOL = {
    "name": "emit_result",
    "description": "输出本 commit 的守门结论。no_issue=true 表示不触及性能面。",
    "input_schema": {"type": "object", "properties": {
        "no_issue": {"type": "boolean"},
        "touches_perf_surface": {"type": "boolean"},
        "findings": {"type": "array", "items": {"type": "object", "properties": {
            "severity": {"type": "string", "enum": ["critical", "important", "suggestion"]},
            "source": {"type": "string", "enum": ["route", "deep-review"]},
            "category": {"type": "string"},
            "file": {"type": ["string", "null"]},
            "claim": {"type": "string"},
            "conf_raw": {"type": "number", "description": "折前置信度（0-1）；harness 施加 unverified 折扣"},
            "unverified": {"type": ["array", "string", "null"], "description": "三问中未核实的项；非空则 harness 打 0.7 折"},
            "suggested_benchmark": {"type": ["string", "null"]}},
            "required": ["severity", "source", "category", "claim", "conf_raw"]}}},
        "required": ["no_issue", "touches_perf_surface", "findings"]},
}

EXCLUDE_RE = re.compile(r"(^|/)(tests?|docs?|examples?|\.github)(/|$)", re.I)
RECIPE_CFG_RE = re.compile(r"(recipe|\.ya?ml$|config|args|arguments|envs?\.py|defaults?)", re.I)

DEFAULT_CONFIG = {
    "version": "detector_v2_1", "budget": 2, "max_turns": 5,
    "large_commit_lines": 400, "large_commit_files": 10,
    "deep_top_k": 2, "scan_block_char_budget": 12000, "unverified_discount": 0.7,
    "formal_severity": ["critical", "important"], "formal_conf_min": 0.5,
    "assets": {"handbook": "knowledge/handbook.v1.jsonl", "repo_profile": "knowledge/repo_profile.v1.json",
               "leaf_verification": "knowledge/leaf_verification.v1.json"},
}


def _cl():
    global _client
    if _client is None:
        from anthropic import AnthropicBedrock
        _client = AnthropicBedrock(aws_region=os.environ.get("AWS_REGION", "us-east-1"), timeout=150, max_retries=4)
    return _client


class Assets:
    def __init__(self, cfg):
        self.handbook = {json.loads(l)["leaf"]: json.loads(l) for l in (BASE / cfg["assets"]["handbook"]).read_text().splitlines()}
        self.profile = json.loads((BASE / cfg["assets"]["repo_profile"]).read_text())
        lv = json.loads((BASE / cfg["assets"]["leaf_verification"]).read_text())
        self.leafverif = {e["leaf"]: e for e in lv["table"]}

    def handbook_page(self, leaf):
        p = self.handbook.get(leaf)
        if not p or p.get("skipped"):
            valid = ", ".join(sorted(self.handbook))
            return f"[无此叶 '{leaf}' 或无手册页]。有效叶名：{valid[:800]}..."
        pg = p.get("page", {})
        return (f"叶 {leaf}:\n反模式: {'; '.join(pg.get('antipatterns', [])[:4])}\n"
                f"检查清单: {'; '.join(pg.get('detection_checklist', [])[:4])}\n"
                f"显现条件: {'; '.join(pg.get('manifest_conditions_top3', [])[:3])}")

    def hot_files(self, repo):
        hf = self.profile["repos"].get(repo, {}).get("hot_files_top50", [])[:30]
        return "\n".join(f"- {h['file']} (top_leaf={h.get('top_leaf')})" for h in hf)

    def bench_for(self, leaf):
        lv = self.leafverif.get(leaf)
        if not lv:
            return None
        if lv.get("verify_kind") == "route-recipe" and lv.get("recipe"):
            return f"{lv['recipe']} — {lv.get('config_hint') or ''}"
        if lv.get("verify_kind") == "microbench":
            return f"microbench_templates/{leaf}.py"
        return None


def gate_predicate(view):
    files = view.get("changed_files", [])
    if not files:
        return False
    return all(EXCLUDE_RE.search(f) for f in files) and not any(RECIPE_CFG_RE.search(f) for f in files)


def _split_blocks(diff):
    chunks = re.split(r"(?=^diff --git )", diff, flags=re.M)
    if len(chunks) <= 1:
        chunks = re.split(r"(?=^\+\+\+ )", diff, flags=re.M)
    return [c for c in chunks if c.strip()]


def make_detect(config=None, ablate=None):
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    ablate = set(ablate or [])
    assets = Assets(cfg)
    route_independent = "route" not in ablate  # CONTRACT B leg; ablation removes it

    def _agentic(view, tools, sys_prompt, block=None, max_turns=None, label="detect"):
        """Run the v2.1 agentic loop; returns (result_dict, cost). result has no_issue/
        touches_perf_surface/findings (conf_raw + unverified)."""
        mt = max_turns or cfg["max_turns"]
        hot = assets.hot_files(view["repo"])
        diff = block if block is not None else view["diff"]
        umsg = (f"COMMIT {view['repo']}@{view['sha'][:12]} (parent {view['parent_sha'][:12]})\n"
                f"=== MESSAGE ===\n{view['commit_message']}\n\n"
                f"=== CHANGED FILES ===\n" + "\n".join(view['changed_files']) + "\n\n"
                f"=== 本仓 top 热点文件（事故高发地图）===\n{hot}\n\n"
                f"=== DIFF ===\n{diff}\n")
        messages = [{"role": "user", "content": [{"type": "text", "text": umsg, "cache_control": CACHE}]}]
        tools_list = [HANDBOOK_TOOL] + DET_TOOLS + [EMIT_TOOL]
        result = None; cost = 0.0
        for turn in range(mt):
            force = turn >= mt - 1
            kw = dict(model=OPUS, max_tokens=2600, thinking={"type": "disabled"},
                      system=[{"type": "text", "text": sys_prompt, "cache_control": CACHE}],
                      messages=messages, tools=tools_list)
            if force:
                kw["tool_choice"] = {"type": "tool", "name": "emit_result"}
            with _cl().messages.stream(**kw) as st:
                resp = st.get_final_message()
            cost += UL.record(_STAGE, label, resp.usage)
            tus = [b for b in resp.content if b.type == "tool_use"]
            emit = next((b for b in tus if b.name == "emit_result"), None)
            if emit is not None:
                result = emit.input; break
            if not tus:
                messages.append({"role": "assistant", "content": resp.content})
                messages.append({"role": "user", "content": "Call emit_result now."}); continue
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for b in tus:
                if b.name == "handbook_lookup":
                    out = assets.handbook_page(b.input.get("leaf", ""))
                else:
                    out = _run_tool(tools, b.name, b.input)
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": out})
            messages.append({"role": "user", "content": results})
        return (result or {"no_issue": True, "touches_perf_surface": False, "findings": []}), cost

    def _finalize(f):
        """Apply CONTRACT C harness discount + attach benchmark. Returns finding with conf_final."""
        conf_raw = float(f.get("conf_raw", f.get("confidence", 0.5)) or 0.5)
        unv = f.get("unverified")
        has_unv = bool(unv) and unv not in ([], "", "none", "None")
        conf_final = round(conf_raw * cfg["unverified_discount"], 4) if has_unv else conf_raw
        f["conf_raw"] = conf_raw
        f["conf_final"] = conf_final
        if not f.get("suggested_benchmark"):
            b = assets.bench_for(f.get("category"))
            if b:
                f["suggested_benchmark"] = b
        return f

    def detect(view, tools):
        meta = {"cost": 0.0, "gate_predicate": gate_predicate(view), "decomposed": False,
                "n_blocks": 1, "skipped_blocks": 0, "scan_truncations": [], "model_step0_no_issue": False,
                "route_invariant_violation": False}
        big = view["diff"].count("\n") > cfg["large_commit_lines"] or len(view.get("changed_files", [])) > cfg["large_commit_files"]
        findings = []
        touches = False
        no_issue = False

        if big:
            # CONTRACT E: cheap scan ALL blocks (batched if long) → top-2 deep at 5 turns
            meta["decomposed"] = True
            blocks = _split_blocks(view["diff"])
            meta["n_blocks"] = len(blocks)
            # cheap suspicion scan, batched to respect char budget (record truncations)
            ranked = _scan_blocks(view, blocks, cfg, meta)
            top = ranked[:cfg["deep_top_k"]]
            meta["skipped_blocks"] = max(0, len(blocks) - len(top))
            # deep review each top block; route decided from the FULL-commit triage (first pass)
            agg_touches = False
            for bi, _score in top:
                res, c = _agentic(view, tools, SYSTEM_V2_1, block=blocks[bi], label="deep-big")
                meta["cost"] += c
                agg_touches = agg_touches or bool(res.get("touches_perf_surface"))
                for f in res.get("findings", []):
                    findings.append(f)
            # a whole-commit triage pass to get route (small prompt, whole message + file list)
            tri, ct = _agentic(view, tools, SYSTEM_V2_1, block="(大 commit：见 CHANGED FILES 与 message 做分诊，深审在其它块进行)", max_turns=2, label="triage-big")
            meta["cost"] += ct
            touches = bool(tri.get("touches_perf_surface")) or agg_touches
            findings += [f for f in tri.get("findings", []) if f.get("source") == "route"]
            no_issue = (not findings) and (not touches)
        else:
            res, c = _agentic(view, tools, SYSTEM_V2_1, label="detect")
            meta["cost"] += c
            touches = bool(res.get("touches_perf_surface"))
            no_issue = bool(res.get("no_issue"))
            meta["model_step0_no_issue"] = no_issue and not touches
            findings = list(res.get("findings", []))

        # CONTRACT B: route independence + invariant enforcement
        has_route = any(f.get("source") == "route" for f in findings)
        if route_independent and touches and not has_route:
            # invariant would be violated → synthesize the mandated route finding (charter B:
            # "叶子不明确不豁免——用最近叶/uncategorized＋conf 0.3"). This keeps the invariant at 0.
            findings.append({"severity": "important", "source": "route", "category": "uncategorized",
                             "claim": "该改动触及性能面（分诊判定 touches=true 但模型未产出 route finding），建议验证",
                             "conf_raw": 0.3, "unverified": ["auto-synthesized route"], "file": None})
            meta["route_synthesized"] = True
            has_route = True
        # invariant metric: touches=true but no route finding (must be 0 when route_independent)
        if route_independent and touches and not has_route:
            meta["route_invariant_violation"] = True

        # ablation: remove the route leg entirely (RUN3 error form — triage picks leaf but no finding)
        if not route_independent:
            findings = [f for f in findings if f.get("source") != "route"]

        # finalize (CONTRACT C discount) + budget=2 by conf_final
        findings = [_finalize(f) for f in findings]
        findings.sort(key=lambda f: f.get("conf_final", 0), reverse=True)
        final = findings[:cfg["budget"]]
        return final, {"n_turns": meta["n_blocks"], **meta, "touches_perf_surface": touches,
                       "no_issue": no_issue, "cost": round(meta["cost"], 5),
                       "tokens": {"in": 0, "out": 0}}
    return detect


def _scan_blocks(view, blocks, cfg, meta):
    """CONTRACT E cheap scan: score each block's suspicion in batches (respect char budget;
    record per-block truncation bytes + total blocks). Returns [(block_idx, score)] desc."""
    budget = cfg["scan_block_char_budget"]
    # batch blocks so each scan call stays within budget; never single-hard-stuff
    batches = []
    cur, cur_len = [], 0
    for i, b in enumerate(blocks):
        bb = b
        if len(b) > budget:
            meta["scan_truncations"].append({"block": i, "orig_bytes": len(b), "kept_bytes": budget})
            bb = b[:budget]
        if cur_len + len(bb) > budget and cur:
            batches.append(cur); cur, cur_len = [], 0
        cur.append((i, bb)); cur_len += len(bb)
    if cur:
        batches.append(cur)
    SCAN_TOOL = {"name": "emit_scores", "description": "为每个块打性能可疑度分(0-1)。",
                 "input_schema": {"type": "object", "properties": {
                     "scores": {"type": "array", "items": {"type": "object", "properties": {
                         "block": {"type": "integer"}, "suspicion": {"type": "number"}}, "required": ["block", "suspicion"]}}},
                     "required": ["scores"]}}
    scores = {}
    for batch in batches:
        txt = "\n\n".join(f"=== BLOCK {i} ===\n{b}" for i, b in batch)
        u = f"大 commit 分块可疑度快扫。为每块给 0-1 性能可疑度（触及计算/通信/显存/调度/服务热路径越高）。\n\n{txt}"
        try:
            with _cl().messages.stream(model=OPUS, max_tokens=800, thinking={"type": "disabled"},
                                       system="你是性能回归快扫器，只打可疑度分，不深究。",
                                       messages=[{"role": "user", "content": u}],
                                       tools=[SCAN_TOOL], tool_choice={"type": "tool", "name": "emit_scores"}) as st:
                r = st.get_final_message()
            UL.record(_STAGE, "scan-big", r.usage)
            inp = next((b.input for b in r.content if b.type == "tool_use"), {}) or {}
            for s in inp.get("scores", []):
                scores[s.get("block")] = s.get("suspicion", 0)
        except Exception:
            for i, _ in batch:
                scores[i] = 0.5
    return sorted(((i, scores.get(i, 0)) for i in range(len(blocks))), key=lambda x: -x[1])


if __name__ == "__main__":
    print("detector_v2_1 module; use run_detector_v2_1.py")
