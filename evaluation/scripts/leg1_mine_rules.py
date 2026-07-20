"""Stage 3 · Leg 1 — static rule mining (RUN2_INSTRUCTIONS.md §3.1).

Mine executable, LLM-free static rules from historical fix patches so the detector
can catch high-detectability regressions cheaply.

Pipeline:
  1. mining set = train cards with static_detectability=high AND kind∈{regression-fix,
     optimization} (~285). Training side MAY look at the fix diff (`git show <card.sha>`).
  2. Opus drafts one rule per card: {id, antipattern, match_logic, positive_ex,
     negative_ex, taxonomy_leaf, matcher_kind}. matcher_kind ∈ {regex, ast} (Semgrep
     optional — we emit regex/ast which run with zero deps).
  3. semantic dedup within a leaf (merge rules with same leaf + overlapping antipattern).
  4. validation is a SEPARATE step (validate_rules in run_rules.py on dev).

Writes rules/ruleset.v1/draft_rules.jsonl. Run AFTER baselines to avoid Bedrock contention.
Usage: uv run python leg1_mine_rules.py --workers 8 --max-cost-usd 60
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mpb_common import BASE, CASES, SPLITS, read_jsonl, now_iso  # noqa: E402
import eval_harness as H  # noqa: E402

OPUS = "us.anthropic.claude-opus-4-8"
PRICE_IN, PRICE_OUT = 5.0, 25.0
RULES_DIR = BASE / "rules" / "ruleset.v1"
_cost_lock = threading.Lock()
_client = None

RULE_TOOL = {
    "name": "emit_rule",
    "description": "Emit one executable static rule distilled from this fix patch, or decline.",
    "input_schema": {"type": "object", "properties": {
        "worth_a_rule": {"type": "boolean",
                         "description": "false if this fix is too case-specific to generalize into a static check"},
        "rule_id": {"type": "string", "description": "short kebab-case id"},
        "antipattern": {"type": "string", "description": "the code smell this catches, one sentence"},
        "matcher_kind": {"type": "string", "enum": ["regex", "ast"]},
        "match_logic": {"type": "string",
                        "description": "for regex: a Python regex matching the ANTIPATTERN (pre-fix) code. "
                                       "for ast: a short predicate description over Python AST nodes."},
        "regex": {"type": ["string", "null"], "description": "the concrete regex if matcher_kind=regex"},
        "positive_example": {"type": "string", "description": "a code line that SHOULD match (the bug)"},
        "negative_example": {"type": "string", "description": "a similar line that should NOT match (benign)"},
        "taxonomy_leaf": {"type": "string"},
        "target_globs": {"type": "array", "items": {"type": "string"},
                         "description": "file globs to restrict the rule, e.g. ['*.py']"},
        "severity": {"type": "string", "enum": ["critical", "important", "suggestion"]},
    }, "required": ["worth_a_rule"]},
}

SYSTEM = """你是性能工程与静态分析专家。给你一个**已确认的性能问题的修复补丁**（fix diff），
你的任务：把"被修复的那个反模式"提炼成一条**可独立执行、不调用 LLM** 的静态规则，
用来在**未来的 diff**里发现同类问题。

关键要求：
- 规则匹配的是**引入 bug 的代码形态（pre-fix / 反模式）**，不是修复后的形态。
- 规则要**可泛化**：不要把具体变量名/文件名写死。抓机制，不抓字面。
- 优先 `regex`（给出真正能用 `re.search` 跑的 Python 正则）；只有当反模式必须靠语法结构
  才能判定时才用 `ast`（描述判定谓词）。
- 如果这个修复太特例、无法泛化成静态规则（比如只是改了一个魔法常数），
  诚实地 `worth_a_rule: false`，不要硬造。
- positive_example 必须能被你的 regex 命中；negative_example 是形似但良性的代码，不应命中。

只调用 `emit_rule` 一次。"""


def _cl():
    global _client
    if _client is None:
        from anthropic import AnthropicBedrock
        _client = AnthropicBedrock(aws_region=os.environ.get("AWS_REGION", "us-east-1"),
                                   timeout=120.0, max_retries=4)
    return _client


def _fix_diff(repo, sha):
    """Training side is allowed to see the fix. Reuse the harness trimmer for consistency."""
    try:
        files = [f for f in H.git(repo, "show", "--first-parent", "--name-only",
                                  "--format=", sha, timeout=90).splitlines() if f.strip()]
        return H._trimmed_diff(repo, sha, files)["text"], files
    except Exception as e:
        return f"[diff error {e}]", []


def mine_one(card):
    repo, sha = card["repo"], card["sha"]
    diff, files = _fix_diff(repo, sha)
    if not diff.strip() or diff.startswith("[diff error"):
        return None, 0.0
    u = (f"FIX commit {repo}@{sha[:12]} — taxonomy leaf: {card.get('taxonomy_label')}\n"
         f"mechanism (ground truth): {card.get('mechanism','')[:400]}\n"
         f"symptom: {card.get('symptom')}\n\n"
         f"=== FIX DIFF (the '+' side removed the bug; distill a rule for the '-'/pre-fix form) ===\n"
         f"{diff[:12000]}")
    with _cl().messages.stream(model=OPUS, max_tokens=1500, thinking={"type": "disabled"},
                               system=SYSTEM, messages=[{"role": "user", "content": u}],
                               tools=[RULE_TOOL],
                               tool_choice={"type": "tool", "name": "emit_rule"}) as st:
        r = st.get_final_message()
    inp = next((b.input for b in r.content if b.type == "tool_use"), None)
    cost = (r.usage.input_tokens * PRICE_IN + r.usage.output_tokens * PRICE_OUT) / 1e6
    if not inp or not inp.get("worth_a_rule"):
        return {"declined": True, "src_case": card["case_id"],
                "leaf": card.get("taxonomy_label")}, cost
    inp["src_case"] = card["case_id"]
    inp["src_repo"] = repo
    return inp, cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-cost-usd", type=float, default=60.0)
    ap.add_argument("--exclude-repo", default=None,
                    help="skip cards from this repo (Stage 5 holdout: --exclude-repo Megatron-LM)")
    a = ap.parse_args()

    cards = {c["case_id"]: c for c in read_jsonl(CASES / "cards_final_v011.jsonl")}
    train = [l.strip() for l in (SPLITS / "train.txt").read_text().splitlines() if l.strip()]
    pool = [cards[l.split(":", 1)[1]] for l in train if l.startswith("case:")
            and cards.get(l.split(":", 1)[1], {}).get("static_detectability") == "high"
            and cards.get(l.split(":", 1)[1], {}).get("kind") in ("regression-fix", "optimization")]
    if a.exclude_repo:
        pool = [c for c in pool if c["repo"] != a.exclude_repo]
    RULES_DIR.mkdir(parents=True, exist_ok=True)
    out = RULES_DIR / ("draft_rules_noMega.jsonl" if a.exclude_repo else "draft_rules.jsonl")
    print(f"leg1 mining pool: {len(pool)} cards (exclude_repo={a.exclude_repo}) → {out}", flush=True)

    state = {"spent": 0.0, "n": 0, "rules": 0, "declined": 0, "stop": False}
    lock = threading.Lock()

    def work(c):
        if state["stop"]:
            return None
        try:
            rec, cost = mine_one(c)
        except Exception as e:
            return {"error": str(e)[:150], "src_case": c["case_id"]}, 0.0
        return rec, cost

    with open(out, "w") as fh, ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(work, c) for c in pool]
        for f in as_completed(futs):
            res = f.result()
            if not res:
                continue
            rec, cost = res
            with _cost_lock:
                state["spent"] += cost; state["n"] += 1
                if rec and rec.get("declined"):
                    state["declined"] += 1
                elif rec and not rec.get("error"):
                    state["rules"] += 1
                if state["spent"] >= a.max_cost_usd:
                    state["stop"] = True
            if rec:
                with lock:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
            if state["n"] % 50 == 0:
                print(f"  {state['n']}/{len(pool)} · {state['rules']} rules · "
                      f"{state['declined']} declined · ${state['spent']:.2f}", flush=True)
    print(f"DONE: {state['n']} mined, {state['rules']} draft rules, {state['declined']} declined, "
          f"~${state['spent']:.2f} → {out}", flush=True)


if __name__ == "__main__":
    main()
