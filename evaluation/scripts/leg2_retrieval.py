"""Stage 3 · Leg 2 — retrieval-augmented review (RUN2_INSTRUCTIONS.md §3.2).

Expected main leg (targets medium-detectability). Two parts:
  build   — compress the 3,386 train perf cards into a KB of
            {case_id, taxonomy_leaf, one-line mechanism, touched APIs/symbols,
             file types, symptom}, and build a dependency-free BM25 index over
            (file paths + diff symbols + APIs). Writes detectors/leg2_kb.jsonl.
  (detect is exposed via make_detect for fusion/ablation.)

Retrieval query = changed file paths + symbol names extracted from the PR-time diff.
Judge = Opus given the PR-time view + top-k historical cases (mechanism + leaf ONLY,
never this item's own label) + the hit leaves' boundary_notes from taxonomy.yaml.

Presentation rule: KB entries are HISTORICAL train cards; the item under test never
contributes its own card. Retrieval uses only the PR-time view (diff/paths/symbols).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import threading
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mpb_common import BASE, CASES, SPLITS, read_jsonl, now_iso  # noqa: E402
import eval_harness as H  # noqa: E402

OPUS = "us.anthropic.claude-opus-4-8"
PRICE_IN, PRICE_OUT = 5.0, 25.0
KB_PATH = BASE / "detectors" / "leg2_kb.jsonl"
_client = None

SYM_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
CALL_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_.]*)\s*\(")


# ---------------------------------------------------------------------------
# dependency-free BM25 (Okapi) over token lists
# ---------------------------------------------------------------------------
class BM25:
    def __init__(self, corpus_tokens, k1=1.5, b=0.75):
        self.docs = corpus_tokens
        self.N = len(corpus_tokens)
        self.avgdl = sum(len(d) for d in corpus_tokens) / max(1, self.N)
        self.k1, self.b = k1, b
        self.df = defaultdict(int)
        self.tf = []
        for d in corpus_tokens:
            c = Counter(d)
            self.tf.append(c)
            for t in c:
                self.df[t] += 1
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in self.df.items()}

    def top(self, query_tokens, k):
        scores = []
        q = [t for t in query_tokens if t in self.idf]
        for i, tf in enumerate(self.tf):
            dl = len(self.docs[i])
            s = 0.0
            for t in q:
                f = tf.get(t, 0)
                if not f:
                    continue
                s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            if s > 0:
                scores.append((s, i))
        scores.sort(reverse=True)
        return scores[:k]


def _symbols_from_diff(view):
    """Extract file paths + symbol/API tokens from the PR-time diff (added+context)."""
    toks = []
    for f in view.get("changed_files", []):
        toks += re.split(r"[/_.]", f)
    added = [l[1:] for l in view["diff"].splitlines() if l.startswith("+") and not l.startswith("+++")]
    body = "\n".join(added)
    toks += CALL_RE.findall(body)
    toks += SYM_RE.findall(body)
    return [t.lower() for t in toks if len(t) >= 3]


def _kb_tokens(entry):
    toks = []
    toks += re.split(r"[/_.]", " ".join(entry.get("file_types", [])))
    toks += [s.lower() for s in entry.get("symbols", [])]
    toks += SYM_RE.findall(entry.get("mechanism", "").lower())
    toks += [entry.get("taxonomy_leaf", "").replace("-", " ")]
    return [t.lower() for t in toks if len(t) >= 3]


def _cl():
    global _client
    if _client is None:
        from anthropic import AnthropicBedrock
        _client = AnthropicBedrock(aws_region=os.environ.get("AWS_REGION", "us-east-1"),
                                   timeout=120.0, max_retries=4)
    return _client


# ---------------------------------------------------------------------------
# build KB
# ---------------------------------------------------------------------------
def cmd_build(a):
    cards = {c["case_id"]: c for c in read_jsonl(CASES / "cards_final_v011.jsonl")}
    train = [l.strip() for l in (SPLITS / "train.txt").read_text().splitlines() if l.strip()]
    perf = [cards[l.split(":", 1)[1]] for l in train if l.startswith("case:")
            and cards.get(l.split(":", 1)[1], {}).get("is_perf_related")]
    if a.exclude_repo:
        perf = [c for c in perf if c["repo"] != a.exclude_repo]
    out = BASE / "detectors" / ("leg2_kb_noMega.jsonl" if a.exclude_repo else "leg2_kb.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    # compress WITHOUT an LLM: mechanism is already one-liner-ish; extract symbols from the
    # card's stored mechanism/evidence (train side may use posterior fields — this is the KB,
    # not the item under test). Symbols come from mechanism text (cheap, deterministic).
    n = 0
    with open(out, "w") as fh:
        for c in perf:
            mech = (c.get("mechanism") or "").strip()
            syms = list(dict.fromkeys(SYM_RE.findall(mech + " " + (c.get("subject") or ""))))[:20]
            entry = {
                "case_id": c["case_id"], "repo": c["repo"],
                "taxonomy_leaf": c.get("taxonomy_label"),
                "mechanism": mech[:300],
                "symptom": c.get("symptom"),
                "symbols": syms,
                "file_types": [],  # filled below from the fix diff paths (train side allowed)
            }
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n"); n += 1
    print(f"leg2 KB: {n} entries → {out} (exclude_repo={a.exclude_repo})", flush=True)


def _boundary_notes():
    """leaf id → boundary_notes text from taxonomy.yaml (best-effort line parse)."""
    notes, cur = {}, None
    for ln in (BASE / "taxonomy" / "taxonomy.yaml").read_text().splitlines():
        m = re.match(r"^\s+- id:\s*(\S+)", ln)
        if m:
            cur = m.group(1)
        bn = re.match(r"^\s+boundary_notes:\s*(.+)", ln)
        if bn and cur:
            notes[cur] = bn.group(1).strip().strip('"\'')
    return notes


REVIEW_TOOL = {
    "name": "emit_findings",
    "description": "Emit perf findings informed by the retrieved historical cases.",
    "input_schema": {"type": "object", "properties": {
        "findings": {"type": "array", "items": {"type": "object", "properties": {
            "severity": {"type": "string", "enum": ["critical", "important", "suggestion"]},
            "category": {"type": "string"},
            "file": {"type": ["string", "null"]},
            "line": {"type": ["integer", "null"]},
            "claim": {"type": "string"},
            "confidence": {"type": "number"},
            "matched_leaf": {"type": ["string", "null"]},
            "suggested_benchmark": {"type": ["string", "null"]}},
            "required": ["severity", "category", "claim", "confidence"]}}},
        "required": ["findings"]},
}

REVIEW_SYS = """你是性能回归审查专家。除了当前 commit 的 PR-time 视图，你还会看到几条**历史上
真实发生过的同类性能问题**（只给机制与分类，不含当前 commit 的任何标签）。把它们当作
"这类改动过去踩过的坑"的先验。

判断当前 diff 是否**引入**了其中某种（或别的）性能退化。命中历史模式时，明确指出是哪一类
机制、在 diff 的哪一处。不要因为"检索到了案例"就强行报——只有当 diff 里确有对应的代码形态
才报。可用 read_file_at_parent / grep_at_parent 到 parent 快照核实调用上下文（冷/热路径）。

通过 emit_findings 输出；matched_leaf 填命中的历史分类叶（若有）。无问题则空列表。"""


def make_detect(kb_path=None, k=5, max_turns=4):
    kb = list(read_jsonl(kb_path or KB_PATH))
    bm = BM25([_kb_tokens(e) for e in kb])
    notes = _boundary_notes()

    def _view_msg(view, retrieved):
        hist = "\n".join(
            f"- [{e['taxonomy_leaf']}] {e['mechanism']} (symptom: {e.get('symptom')})"
            for e in retrieved)
        leaves = {e["taxonomy_leaf"] for e in retrieved}
        bnotes = "\n".join(f"- {lf}: {notes[lf]}" for lf in leaves if lf in notes)
        return (f"COMMIT {view['repo']}@{view['sha'][:12]} (parent {view['parent_sha'][:12]})\n"
                f"=== MESSAGE ===\n{view['commit_message']}\n\n"
                f"=== CHANGED FILES ===\n" + "\n".join(view['changed_files']) + "\n\n"
                f"=== DIFF ===\n{view['diff']}\n\n"
                f"=== 检索到的历史同类性能问题（先验，非本 commit 标签）===\n{hist or '(none)'}\n\n"
                f"=== 命中类别的边界说明 ===\n{bnotes or '(none)'}\n")

    from detectors_baseline import DET_TOOLS, _run_tool, CACHE, _mark_last_cache

    def detect(view, tools):
        q = _symbols_from_diff(view)
        hits = bm.top(q, k)
        retrieved = [kb[i] for _, i in hits]
        messages = [{"role": "user", "content": _view_msg(view, retrieved)}]
        _mark_last_cache(messages)
        itok = otok = cw = cr = 0
        findings = None
        turns = 0
        for turn in range(max_turns):
            turns = turn + 1
            force = turn >= max_turns - 1
            kw = dict(model=OPUS, max_tokens=2500, thinking={"type": "disabled"},
                      system=[{"type": "text", "text": REVIEW_SYS, "cache_control": CACHE}],
                      messages=messages, tools=DET_TOOLS + [REVIEW_TOOL])
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
                findings = emit.input.get("findings", [])
                break
            if not tus:
                messages.append({"role": "assistant", "content": resp.content})
                messages.append({"role": "user", "content": "Call emit_findings now."})
                _mark_last_cache(messages)
                continue
            messages.append({"role": "assistant", "content": resp.content})
            results = [{"type": "tool_result", "tool_use_id": b.id,
                        "content": _run_tool(tools, b.name, b.input)} for b in tus]
            messages.append({"role": "user", "content": results})
            _mark_last_cache(messages)
        cost = (itok * PRICE_IN + otok * PRICE_OUT + cw * 6.25 + cr * 0.5) / 1e6
        return (findings or []), {"n_turns": turns, "k": k,
                                  "retrieved_leaves": [e["taxonomy_leaf"] for e in retrieved],
                                  "tokens": {"in": itok, "out": otok, "cache_w": cw, "cache_r": cr},
                                  "cost": round(cost, 5)}
    return detect


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--exclude-repo", default=None)
    a = ap.parse_args()
    if a.cmd == "build":
        cmd_build(a)


if __name__ == "__main__":
    main()
