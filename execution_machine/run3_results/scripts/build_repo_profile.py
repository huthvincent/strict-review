"""RUN3 Stage 2.2 — Repo-Profiler v1 (knowledge/repo_profile.v1.json), train-only, deterministic.

Per repo (4): hot-file leaderboard (train perf cards × commit files, top50, with top leaf) /
module→susceptible-leaf table / exclusion table (tests|docs|examples|.github) /
Megatron section adds recipe-coverage hint (one LLM inference over recipe configs, logged).

Hot files: for each train perf card, list the files it touched (git show --name-only at the
card sha) and attribute to the card's leaf. Aggregate over train only.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mpb_common import BASE, CASES, SPLITS, read_jsonl, now_iso  # noqa: E402
import eval_harness as H  # noqa: E402

REPOS = ["Megatron-LM", "vllm", "DeepSpeed", "TransformerEngine"]
EXCLUDE_RE = re.compile(r"(^|/)(tests?|docs?|examples?|\.github)(/|$)", re.I)
KDIR = BASE / "knowledge"


def _files_at(repo, sha):
    try:
        out = H.git(repo, "show", "--first-parent", "--name-only", "--format=", sha, timeout=60)
        return [f for f in out.splitlines() if f.strip()]
    except Exception:
        return []


def _module_of(path):
    """Coarse module = first 2 path components (dir)."""
    parts = path.split("/")
    return "/".join(parts[:2]) if len(parts) >= 2 else (parts[0] if parts else "?")


def main():
    KDIR.mkdir(parents=True, exist_ok=True)
    cards = {c["case_id"]: c for c in read_jsonl(CASES / "cards_final_v011.jsonl")}
    train = set(l.strip().split(":", 1)[1] for l in (SPLITS / "train.txt").read_text().splitlines()
                if l.strip().startswith("case:"))
    perf = [c for cid, c in cards.items() if cid in train and c.get("is_perf_related")]

    prof = {"generated": now_iso(), "source": "train perf cards only", "repos": {}}
    for repo in REPOS:
        rcards = [c for c in perf if c["repo"] == repo]
        file_ct = Counter()
        file_leaf = defaultdict(Counter)
        mod_leaf = defaultdict(Counter)
        for c in rcards:
            files = [f for f in _files_at(repo, c["sha"]) if not EXCLUDE_RE.search(f)]
            leaf = c.get("taxonomy_label")
            for f in files:
                file_ct[f] += 1
                if leaf:
                    file_leaf[f][leaf] += 1
                    mod_leaf[_module_of(f)][leaf] += 1
        hot = []
        for f, n in file_ct.most_common(50):
            topleaf = file_leaf[f].most_common(1)
            hot.append({"file": f, "n_cards": n, "top_leaf": topleaf[0][0] if topleaf else None})
        mod_table = {}
        for m, lc in sorted(mod_leaf.items(), key=lambda x: -sum(x[1].values()))[:20]:
            mod_table[m] = [{"leaf": l, "n": n} for l, n in lc.most_common(3)]
        prof["repos"][repo] = {
            "n_train_perf_cards": len(rcards),
            "hot_files_top50": hot,
            "module_to_susceptible_leaves_top20": mod_table,
            "exclusion_patterns": ["tests/", "docs/", "examples/", ".github/"],
        }
        print(f"{repo}: {len(rcards)} cards, {len(file_ct)} distinct hot files", flush=True)

    # Megatron recipe-coverage hint (one LLM inference, logged) — best-effort, non-blocking
    mega_recipe_note = _megatron_recipe_hint()
    if mega_recipe_note:
        prof["repos"]["Megatron-LM"]["recipe_coverage_hint"] = mega_recipe_note

    (KDIR / "repo_profile.v1.json").write_text(json.dumps(prof, ensure_ascii=False, indent=2))
    print(f"wrote knowledge/repo_profile.v1.json", flush=True)


def _megatron_recipe_hint():
    """One Opus inference mapping existing Megatron perf recipes → which leaves they'd exercise.
    Logged (human-readable). Non-blocking: returns None on any failure."""
    try:
        import glob
        rd = os.environ.get("MPB_REPOS", "/home/ec2-user/megaperf_repos") + "/Megatron-LM"
        recs = sorted(set(os.path.basename(p).replace(".yaml", "")
                          for p in glob.glob(rd + "/tests/test_utils/recipes/**/*perf*.yaml", recursive=True)))
        if not recs:
            return None
        from anthropic import AnthropicBedrock
        cl = AnthropicBedrock(aws_region=os.environ.get("AWS_REGION", "us-east-1"), timeout=90, max_retries=3)
        u = ("Megatron-LM 现有性能 CI recipe（文件名）如下：\n" + ", ".join(recs) +
             "\n\n简述每个 recipe 大致覆盖哪类性能敏感面（1 行），供路由参考。纯依据名称与常识，勿编造具体数字。")
        r = cl.messages.create(model="us.anthropic.claude-opus-4-8", max_tokens=600,
                               messages=[{"role": "user", "content": u}])
        txt = "".join(b.text for b in r.content if b.type == "text")
        return {"recipes_found": recs, "llm_hint": txt,
                "provenance": {"model": "us.anthropic.claude-opus-4-8", "ts": now_iso()}}
    except Exception as e:
        return {"error": str(e)[:120]}


if __name__ == "__main__":
    main()
