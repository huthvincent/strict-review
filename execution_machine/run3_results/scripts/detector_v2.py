"""RUN3 Stage 3 — detector_v2 (§3). Config frozen to detector/detector_v2.config.json.

Pipeline per item (PR-time view only; leak-safe via harness Tools):
  3.1 PROFILE GATE: no_issue ONLY if ALL changed files fall in exclude dirs
      (tests|docs|examples|.github) AND diff touches no recipe/config-file path.
  3.2 LARGE-COMMIT DECOMPOSITION: diff>400 lines or >10 files → group by file, review
      each group, merge+rank.
  3.3 MAIN REVIEW: handbook resident (trimmed to relevant 10-20 leaves via profile/initial
      classification) + DET_TOOLS parent-snapshot verification; prompt hard-codes THREE
      questions (hot path? caller? default-enabled?). Prefer precision (宁缺毋滥).
  3.4 leg3 ROUTING preserved: classify→leaf→leaf_verification → finding gets
      suggested_benchmark or template name.
  3.5 NO adversarial layer (RUN2 ablation). budget=2 by confidence.

Handbook/profile/leaf_verification are TRAIN-only assets (Stage 2), loaded read-only.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mpb_common import BASE, now_iso  # noqa: E402
import eval_harness as H  # noqa: E402
from detectors_baseline import DET_TOOLS, _run_tool, CACHE  # noqa: E402

OPUS = "us.anthropic.claude-opus-4-8"
PRICE_IN, PRICE_OUT = 5.0, 25.0
KDIR = BASE / "knowledge"
CONFIG_PATH = BASE / "detector" / "detector_v2.config.json"
_client = None

EXCLUDE_RE = re.compile(r"(^|/)(tests?|docs?|examples?|\.github)(/|$)", re.I)
# recipe/config file paths that must NOT be gated even if in an excluded dir
RECIPE_CFG_RE = re.compile(r"(recipe|\.ya?ml$|config|args|arguments|envs?\.py|defaults?)", re.I)

DEFAULT_CONFIG = {
    "version": "detector_v2",
    "large_commit_lines": 400, "large_commit_files": 10,
    "decomp_max_groups": 6, "decomp_max_turns": 2,
    "handbook_leaves_min": 10, "handbook_leaves_max": 20,
    "budget": 2, "max_turns": 5, "adversarial": False,
    "assets": {"handbook": "knowledge/handbook.v1.jsonl",
               "repo_profile": "knowledge/repo_profile.v1.json",
               "leaf_verification": "knowledge/leaf_verification.v1.json"},
    "gate": {"exclude_dirs": ["tests", "docs", "examples", ".github"],
             "recipe_cfg_regex": RECIPE_CFG_RE.pattern},
}

FINDINGS_TOOL = {
    "name": "emit_findings",
    "description": "Emit perf findings for this commit (empty if none). 宁缺毋滥。",
    "input_schema": {"type": "object", "properties": {
        "findings": {"type": "array", "items": {"type": "object", "properties": {
            "severity": {"type": "string", "enum": ["critical", "important", "suggestion"]},
            "category": {"type": "string"},
            "leaf": {"type": ["string", "null"]},
            "file": {"type": ["string", "null"]},
            "claim": {"type": "string"},
            "confidence": {"type": "number"},
            "hot_path": {"type": ["boolean", "null"]},
            "default_enabled": {"type": ["boolean", "null"]}},
            "required": ["severity", "category", "claim", "confidence"]}}},
        "required": ["findings"]},
}

CLASSIFY_TOOL = {
    "name": "route_leaves",
    "description": "Pick the 10-20 taxonomy leaves most relevant to this diff (for handbook trimming).",
    "input_schema": {"type": "object", "properties": {
        "leaves": {"type": "array", "items": {"type": "string"}},
        "scenario": {"type": "string", "enum": ["high", "medium", "low", "none"]}},
        "required": ["leaves"]},
}


def _cl():
    global _client
    if _client is None:
        from anthropic import AnthropicBedrock
        _client = AnthropicBedrock(aws_region=os.environ.get("AWS_REGION", "us-east-1"),
                                   timeout=150, max_retries=4)
    return _client


class Assets:
    def __init__(self, cfg):
        self.handbook = {}
        for l in (BASE / cfg["assets"]["handbook"]).read_text().splitlines():
            r = json.loads(l)
            self.handbook[r["leaf"]] = r
        self.profile = json.loads((BASE / cfg["assets"]["repo_profile"]).read_text())
        lv = json.loads((BASE / cfg["assets"]["leaf_verification"]).read_text())
        self.leafverif = {e["leaf"]: e for e in lv["table"]}
        self.all_leaves = sorted(self.handbook.keys())

    def leaf_page_md(self, leaf):
        p = self.handbook.get(leaf, {})
        if p.get("skipped"):
            return f"### {leaf}: (无 train 卡，跳过)"
        pg = p.get("page", {})
        return (f"### {leaf}\n"
                f"反模式: {'; '.join(pg.get('antipatterns', [])[:3])}\n"
                f"检查: {'; '.join(pg.get('detection_checklist', [])[:3])}\n"
                f"显现: {'; '.join(pg.get('manifest_conditions_top3', [])[:2])}")


# ---- 3.1 profile gate ----
def profile_gate(view, cfg):
    """Return True if the commit should be directly no_issue (all excluded + no recipe/config)."""
    files = view.get("changed_files", [])
    if not files:
        return False
    all_excluded = all(EXCLUDE_RE.search(f) for f in files)
    touches_recipe_cfg = any(RECIPE_CFG_RE.search(f) for f in files)
    return all_excluded and not touches_recipe_cfg


# ---- 3.2 large-commit split ----
def _diff_line_count(view):
    return view.get("diff", "").count("\n")


def _split_diff_by_file(view):
    """Split a diff into per-file chunks (best-effort by 'diff --git' / '+++' markers)."""
    diff = view["diff"]
    chunks = re.split(r"(?=^diff --git )", diff, flags=re.M)
    if len(chunks) <= 1:
        chunks = re.split(r"(?=^\+\+\+ )", diff, flags=re.M)
    return [c for c in chunks if c.strip()]


SYS_MAIN = """你是 AI 基础设施性能回归审查专家。给你一个 commit 的 PR-time 视图（diff/消息/
父快照工具）+ **相关叶子的检测手册页**（从 train 蒸馏）。判断这个改动是否**引入**了运行时
性能退化（吞吐/延迟/显存/GPU 利用率/挂起）。

**每条候选 finding 必须回答三问**（写进 finding 字段）：
1. 热路径？（每 step/每 microbatch/每参数循环里，还是一次性初始化/冷路径）
2. 调用方？（用 read_file_at_parent/grep_at_parent 到父快照核实这段代码被谁调用、多频繁）
3. 默认配置启用？（该退化路径在默认配置下是否会走到，还是要 opt-in flag）

**宁缺毋滥**：只有三问都指向真实退化才报 important/critical；否则不报或降为 suggestion。
通过 emit_findings 输出；leaf 填命中的 taxonomy 叶。无问题 → 空列表。"""


def _view_msg(view, handbook_md):
    return (f"COMMIT {view['repo']}@{view['sha'][:12]} (parent {view['parent_sha'][:12]})\n"
            f"=== MESSAGE ===\n{view['commit_message']}\n\n"
            f"=== CHANGED FILES ===\n" + "\n".join(view['changed_files']) + "\n\n"
            f"=== DIFF ===\n{view['diff']}\n\n"
            f"=== 相关叶子手册（train 蒸馏，先验）===\n{handbook_md}\n")


def make_detect(config=None, ablate=None):
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    ablate = set(ablate or [])
    assets = Assets(cfg)
    use_handbook = "handbook" not in ablate
    use_tools = "tools" not in ablate  # parent-snapshot DET_TOOLS

    def _route_leaves(view):
        """Light classification: pick relevant leaves (handbook trimming). Uses profile hints."""
        repo = view["repo"]
        prof = assets.profile["repos"].get(repo, {})
        # seed candidate leaves from hot-file/module hints matching changed files
        cand = set()
        files = view.get("changed_files", [])
        for hf in prof.get("hot_files_top50", []):
            if hf["file"] in files and hf.get("top_leaf"):
                cand.add(hf["top_leaf"])
        for mod, leaves in prof.get("module_to_susceptible_leaves_top20", {}).items():
            if any(f.startswith(mod) for f in files):
                for x in leaves:
                    cand.add(x["leaf"])
        # ask Opus to finalize 10-20 leaves given the diff + candidates
        u = (f"改动文件: {files}\n候选叶(来自仓库画像): {sorted(cand)}\n\n"
             f"DIFF(前3000字):\n{view['diff'][:3000]}\n\n"
             f"从这 74 叶中挑 {cfg['handbook_leaves_min']}-{cfg['handbook_leaves_max']} 个最相关的叶用于详审。"
             f"全部叶: {assets.all_leaves}")
        try:
            with _cl().messages.stream(model=OPUS, max_tokens=400, thinking={"type": "disabled"},
                                       system="你是性能回归分诊器，只挑相关 taxonomy 叶，不下结论。",
                                       messages=[{"role": "user", "content": u}],
                                       tools=[CLASSIFY_TOOL],
                                       tool_choice={"type": "tool", "name": "route_leaves"}) as st:
                r = st.get_final_message()
            inp = next((b.input for b in r.content if b.type == "tool_use"), {}) or {}
            leaves = [l for l in inp.get("leaves", []) if l in assets.handbook][:cfg["handbook_leaves_max"]]
            cost = (r.usage.input_tokens * PRICE_IN + r.usage.output_tokens * PRICE_OUT) / 1e6
            return (leaves or sorted(cand)[:cfg["handbook_leaves_max"]], inp.get("scenario", "medium"), cost)
        except Exception:
            return (sorted(cand)[:cfg["handbook_leaves_max"]], "medium", 0.0)

    def _review(view, tools, handbook_md, max_turns=None):
        """One review pass over a (possibly partial) view. Returns (findings, cost).
        max_turns override lets decomposed sub-group reviews use a tighter cap (narrow diffs)."""
        mt = max_turns or cfg["max_turns"]
        messages = [{"role": "user", "content": _view_msg(view, handbook_md)}]
        # cache the (large, stable) handbook+system prefix
        messages[0]["content"] = [{"type": "text", "text": messages[0]["content"], "cache_control": CACHE}]
        itok = otok = cw = cr = 0
        findings = None
        tools_list = (DET_TOOLS if use_tools else []) + [FINDINGS_TOOL]
        for turn in range(mt):
            force = turn >= mt - 1
            kw = dict(model=OPUS, max_tokens=2500, thinking={"type": "disabled"},
                      system=[{"type": "text", "text": SYS_MAIN, "cache_control": CACHE}],
                      messages=messages, tools=tools_list)
            if force:
                kw["tool_choice"] = {"type": "tool", "name": "emit_findings"}
            with _cl().messages.stream(**kw) as st:
                resp = st.get_final_message()
            u = resp.usage
            itok += u.input_tokens; otok += u.output_tokens
            cw += getattr(u, "cache_creation_input_tokens", 0) or 0
            cr += getattr(u, "cache_read_input_tokens", 0) or 0
            tus = [b for b in resp.content if b.type == "tool_use"]
            emit = next((b for b in tus if b.name == "emit_findings"), None)
            if emit is not None:
                findings = emit.input.get("findings", []); break
            if not tus:
                messages.append({"role": "assistant", "content": resp.content})
                messages.append({"role": "user", "content": "Call emit_findings now."}); continue
            messages.append({"role": "assistant", "content": resp.content})
            results = [{"type": "tool_result", "tool_use_id": b.id,
                        "content": _run_tool(tools, b.name, b.input)} for b in tus if b.name != "emit_findings"]
            messages.append({"role": "user", "content": results})
        cost = (itok * PRICE_IN + otok * PRICE_OUT + cw * 6.25 + cr * 0.5) / 1e6
        return (findings or []), cost

    def _attach_benchmark(f):
        """3.4 leg3 routing: attach suggested_benchmark / template from leaf_verification."""
        leaf = f.get("leaf")
        lv = assets.leafverif.get(leaf)
        if not lv:
            return f
        if lv["verify_kind"] == "route-recipe" and lv.get("recipe"):
            f["suggested_benchmark"] = f"{lv['recipe']} — {lv.get('config_hint') or ''}"
        elif lv["verify_kind"] == "microbench":
            f["suggested_benchmark"] = f"microbench_templates/{leaf}.py"
        return f

    def detect(view, tools):
        meta = {"cost": 0.0, "gate": False, "decomposed": False, "n_groups": 1, "scenario": None}
        # 3.1 gate
        if profile_gate(view, cfg):
            meta["gate"] = True
            return [], {"n_turns": 0, "cost": 0.0, **meta, "tokens": {"in": 0, "out": 0}}
        # route leaves for handbook trimming
        if use_handbook:
            leaves, scenario, rcost = _route_leaves(view)
            meta["cost"] += rcost; meta["scenario"] = scenario
            handbook_md = "\n".join(assets.leaf_page_md(l) for l in leaves) if leaves else "(无相关叶)"
        else:
            handbook_md = "(手册消融：本次不提供手册)"
        # 3.2 large-commit decomposition
        big = _diff_line_count(view) > cfg["large_commit_lines"] or len(view.get("changed_files", [])) > cfg["large_commit_files"]
        all_f = []
        if big:
            meta["decomposed"] = True
            chunks = _split_diff_by_file(view)
            meta["n_groups"] = len(chunks)
            cap = cfg.get("decomp_max_groups", 6)
            dturns = cfg.get("decomp_max_turns", 2)  # narrow sub-diffs need fewer turns
            for ch in chunks[:cap]:
                sub = {**view, "diff": ch}
                fs, c = _review(sub, tools, handbook_md, max_turns=dturns)
                meta["cost"] += c
                all_f += fs
        else:
            fs, c = _review(view, tools, handbook_md)
            meta["cost"] += c
            all_f += fs
        # 3.4 attach benchmark + dedup by (file, leaf) + budget trim by confidence
        for f in all_f:
            _attach_benchmark(f)
        best = {}
        for f in all_f:
            k = (f.get("file"), f.get("leaf"), (f.get("claim") or "")[:40])
            if k not in best or f.get("confidence", 0) > best[k].get("confidence", 0):
                best[k] = f
        merged = list(best.values())
        sev = [f for f in merged if f.get("severity") in ("critical", "important")]
        sev.sort(key=lambda f: ({"critical": 3, "important": 2}.get(f.get("severity"), 0), f.get("confidence", 0)), reverse=True)
        rest = sorted([f for f in merged if f not in sev], key=lambda f: f.get("confidence", 0), reverse=True)
        final = (sev + rest)[:cfg["budget"]]
        return final, {"n_turns": meta["n_groups"], "cost": round(meta["cost"], 5), **meta,
                       "tokens": {"in": 0, "out": 0}}
    return detect


if __name__ == "__main__":
    print("detector_v2 module; use run_detector_v2.py / detector runner")
