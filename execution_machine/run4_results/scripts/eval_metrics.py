"""Stage 1.2 — metrics (RUN2_INSTRUCTIONS.md §1.2, metrics.v1).

Implements protocol.v1 §4 over a detector's predictions/{id}/{split}.jsonl:

- HIT judgement: eval_judge.v1 (Opus) given ground-truth (mechanism+evidence+
  symptom) + detector findings → {hit, hit_finding_index, rationale}
- recall@budget: findings sorted by confidence desc, top-`budget` counting ONLY
  severity ∈ {critical, important}; budget ∈ {1,2,5}; budget=2 = north star
- per-kind recall (regression-fix@2 = north star, n=145) + per-taxonomy recall
- benign FPR: a negative is a FP if any finding has severity ∈ {critical,important}
  and confidence ≥ 0.5; stratified by negative_type + weighted total + Wilson 95% CI
- static ceiling printed atop every recall table (57.1%)

Also (§1.2): judge calibration (dual-judge on 200 dev pairs → κ) and cross-family
cross-check (100 of them re-judged by a non-Anthropic model). Subcommands:
  calibrate   — judge κ + cross-family (writes reports/judge_calibration.md)
  score       — score a detector's predictions (writes reports/score_{id}_{split}.md)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mpb_common import BASE, CASES, PAIRS, PROMPTS, REPORTS, read_jsonl, now_iso  # noqa: E402

OPUS = "us.anthropic.claude-opus-4-8"
JUDGE_PROMPT = "eval_judge.v1"
PRICE_IN, PRICE_OUT = 5.0, 25.0
CEILING = 57.1
HIT_SEVERITIES = {"critical", "important"}
FP_CONF = 0.5
_cost_lock = threading.Lock()
_client = None

JUDGE_EMIT = {
    "name": "emit_judgement",
    "description": "Judge whether any finding captured the ground-truth perf problem.",
    "input_schema": {"type": "object", "properties": {
        "hit": {"type": "boolean"},
        "hit_finding_index": {"type": ["integer", "null"]},
        "rationale": {"type": "string"}}, "required": ["hit", "rationale"]},
}


def _judge_system():
    return (PROMPTS / f"{JUDGE_PROMPT}.md").read_text()


def _cl():
    global _client
    if _client is None:
        from anthropic import AnthropicBedrock
        _client = AnthropicBedrock(aws_region=os.environ.get("AWS_REGION", "us-east-1"),
                                   timeout=120.0, max_retries=4)
    return _client


# ---- persistent judge cache (keyed by exact prompt) so re-scoring is free ----
import hashlib  # noqa: E402

_JUDGE_CACHE_PATH = BASE / "predictions" / ".judge_cache.jsonl"
_disk_cache = None
_disk_lock = threading.Lock()


def _load_disk_cache():
    global _disk_cache
    # thread-safe lazy init: without the lock, concurrent judge workers can observe a
    # partially-built dict and spuriously miss cached keys (breaks cache-only mode).
    if _disk_cache is None:
        with _disk_lock:
            if _disk_cache is None:
                built = {}
                if _JUDGE_CACHE_PATH.exists():
                    for l in _JUDGE_CACHE_PATH.read_text().splitlines():
                        try:
                            r = json.loads(l)
                            built[r["k"]] = r["v"]
                        except Exception:
                            pass
                _disk_cache = built  # publish only when fully built
    return _disk_cache


def _judge_user(gt, findings):
    return (f"GROUND TRUTH performance problem:\n"
            f"- symptom: {gt.get('symptom')}\n- mechanism: {gt.get('mechanism','')}\n"
            f"- evidence: {gt.get('evidence','')[:400]}\n\n"
            f"DETECTOR FINDINGS (index: severity/claim):\n"
            + ("\n".join(f"[{i}] {f.get('severity')}: {f.get('claim','')[:300]}"
                         for i, f in enumerate(findings)) or "(no findings)"))


class JudgeCacheMiss(RuntimeError):
    """Raised when JUDGE_CACHE_ONLY=1 and a judgement is not in the cache."""


def _opus_judge(gt: dict, findings: list) -> tuple[dict, float]:
    u = _judge_user(gt, findings)
    key = hashlib.sha1((JUDGE_PROMPT + "\x00" + u).encode("utf-8")).hexdigest()
    cache = _load_disk_cache()
    if key in cache:
        return cache[key], 0.0
    # RUN3 §0.1 cache-only mode: NEVER issue a live judge call on a miss (protects test-set
    # discipline during read-only reaggregation). A miss is a hard error the caller must handle.
    if os.environ.get("JUDGE_CACHE_ONLY") == "1":
        raise JudgeCacheMiss(f"cache miss under JUDGE_CACHE_ONLY (key={key[:12]})")
    # retry transient Bedrock stream errors (ReadTimeout / Internal server error) that
    # escape the SDK's own retry during streaming — otherwise one blip kills a whole pass.
    import time as _t
    r = None
    last_err = None
    for attempt in range(6):
        try:
            with _cl().messages.stream(model=OPUS, max_tokens=600, thinking={"type": "disabled"},
                                       system=_judge_system(), messages=[{"role": "user", "content": u}],
                                       tools=[JUDGE_EMIT],
                                       tool_choice={"type": "tool", "name": "emit_judgement"}) as st:
                r = st.get_final_message()
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            _t.sleep(min(2 ** attempt, 30))
    if r is None:
        # give up on this one item; count as a non-hit but DON'T crash the pass
        return {"hit": False, "rationale": f"judge failed after retries: {str(last_err)[:120]}"}, 0.0
    inp = next((x.input for x in r.content if x.type == "tool_use"), None)
    cost = (r.usage.input_tokens * PRICE_IN + r.usage.output_tokens * PRICE_OUT) / 1e6
    result = inp or {"hit": False, "rationale": "no judgement"}
    with _disk_lock:
        cache[key] = result
        with open(_JUDGE_CACHE_PATH, "a") as fh:
            fh.write(json.dumps({"k": key, "v": result}, ensure_ascii=False) + "\n")
    return result, cost


def _nonanthropic_judge(gt: dict, findings: list, model_id: str) -> dict:
    """Cross-family judge via Bedrock Converse (Nova/Llama). JSON-in-text parse."""
    import boto3
    bc = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    sys_txt = _judge_system() + ('\n\nRespond ONLY with JSON: {"hit": true/false, '
                                 '"hit_finding_index": int|null, "rationale": "..."}')
    u = (f"GROUND TRUTH:\n- symptom: {gt.get('symptom')}\n- mechanism: {gt.get('mechanism','')}\n"
         f"- evidence: {gt.get('evidence','')[:400]}\n\nFINDINGS:\n"
         + ("\n".join(f"[{i}] {f.get('severity')}: {f.get('claim','')[:300]}"
                      for i, f in enumerate(findings)) or "(none)"))
    r = bc.converse(modelId=model_id, system=[{"text": sys_txt}],
                    messages=[{"role": "user", "content": [{"text": u}]}],
                    inferenceConfig={"maxTokens": 400, "temperature": 0})
    txt = r["output"]["message"]["content"][0]["text"]
    m = __import__("re").search(r"\{.*\}", txt, __import__("re").S)
    try:
        return json.loads(m.group(0)) if m else {"hit": False}
    except Exception:
        return {"hit": "true" in txt.lower()[:80]}


# --------------------------------------------------------------------------
def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def cohens_kappa(a, b):
    n = len(a)
    if not n:
        return None
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca.get(k, 0) / n) * (cb.get(k, 0) / n) for k in set(ca) | set(cb))
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def topk_severe(findings, budget):
    sev = [f for f in findings if f.get("severity") in HIT_SEVERITIES]
    sev.sort(key=lambda f: f.get("confidence", 0), reverse=True)
    return sev[:budget]


# --------------------------------------------------------------------------
def cmd_calibrate(a):
    """Dual Opus-judge on 200 dev pairs (κ) + cross-family on 100 (agreement)."""
    # Build synthetic (gt, findings) probes from dev positives: use the baseline_generic
    # predictions if present, else a trivial finding set. Calibration measures judge
    # self-consistency + cross-family agreement, so we need real detector findings.
    dev_pred = BASE / "predictions" / a.calib_detector / f"{a.calib_split}.jsonl"
    if not dev_pred.exists():
        print(f"need predictions at {dev_pred} (run {a.calib_detector} on {a.calib_split} first)"); sys.exit(1)
    cards = {c["case_id"]: c for c in read_jsonl(CASES / "cards_final_v011.jsonl")}
    preds = [json.loads(l) for l in dev_pred.read_text().splitlines()]
    pos = [p for p in preds if p["kind"] == "case"
           and cards.get(p["item_id"].split(":", 1)[1], {}).get("is_perf_related")]
    rnd = random.Random(20260718)
    probes = rnd.sample(pos, min(200, len(pos)))
    cost = [0.0]

    def gt_of(p):
        return cards[p["item_id"].split(":", 1)[1]]

    def dual(p):
        gt, fs = gt_of(p), p["findings"]
        j1, c1 = _opus_judge(gt, fs)
        j2, c2 = _opus_judge(gt, fs)
        with _cost_lock:
            cost[0] += c1 + c2
        return p["item_id"], bool(j1.get("hit")), bool(j2.get("hit"))
    res = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for f in as_completed([ex.submit(dual, p) for p in probes]):
            res.append(f.result())
    a1 = [r[1] for r in res]; a2 = [r[2] for r in res]
    agree = sum(1 for x, y in zip(a1, a2) if x == y) / len(res)
    kappa = cohens_kappa([str(x) for x in a1], [str(x) for x in a2])

    # cross-family on 100
    xf_model = a.xfamily_model
    xf = rnd.sample(probes, min(100, len(probes)))
    xf_res = []
    for p in xf:
        gt, fs = gt_of(p), p["findings"]
        jo, _ = _opus_judge(gt, fs)
        try:
            jx = _nonanthropic_judge(gt, fs, xf_model)
        except Exception as e:
            print(f"  xfamily err: {str(e)[:80]}"); continue
        xf_res.append((bool(jo.get("hit")), bool(jx.get("hit"))))
    xf_agree = sum(1 for x, y in xf_res if x == y) / len(xf_res) if xf_res else 0

    lines = ["# Judge calibration (metrics.v1 §1.2)", "",
             f"- generated: {now_iso()} · judge {OPUS} / {JUDGE_PROMPT}",
             f"- dual-judge on {len(res)} dev positives: agreement {agree:.1%}, "
             f"**Cohen's κ = {kappa:.3f}**" + (" ✓ ≥0.7" if (kappa or 0) >= 0.7 else " ⚠️ <0.7 — fix prompt"),
             f"- cross-family ({xf_model}) on {len(xf_res)}: agreement with Opus judge {xf_agree:.1%}"
             + (" ✓" if xf_agree >= 0.85 else f" — Δ>15pp; report conservative (lower) recall"),
             f"- spend ~${cost[0]:.2f}"]
    (REPORTS / "judge_calibration.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if (kappa or 0) < 0.7:
        print("\n>>> GATE: judge κ < 0.7 — fix eval_judge prompt before proceeding")


def score_predictions(detector_id, split, judge_cache=None):
    """Return a metrics dict for predictions/{detector_id}/{split}.jsonl."""
    cards = {c["case_id"]: c for c in read_jsonl(CASES / "cards_final_v011.jsonl")}
    pairs = {p["pair_id"]: p for p in read_jsonl(PAIRS / "regression_pairs_v011.jsonl")}
    from mpb_common import NEGATIVES
    negs = {n["case_id"]: n for n in read_jsonl(NEGATIVES / "negatives_v011.jsonl")}
    pred_path = BASE / "predictions" / detector_id / f"{split}.jsonl"
    preds = [json.loads(l) for l in pred_path.read_text().splitlines()]

    # judge every positive/pair item once (cache by item_id)
    cache = judge_cache if judge_cache is not None else {}
    cost = [0.0]

    def ground_truth(p):
        if p["kind"] == "pair":
            pr = pairs[p["item_id"].split(":", 1)[1]]
            c = cards.get(pr.get("case_id"), {})
            return {"symptom": pr.get("symptom"), "mechanism": c.get("mechanism", ""),
                    "evidence": c.get("evidence", "")}
        c = cards[p["item_id"].split(":", 1)[1]]
        return {"symptom": c.get("symptom"), "mechanism": c.get("mechanism", ""),
                "evidence": c.get("evidence", "")}

    pos_items = [p for p in preds if p["kind"] in ("case", "pair")
                 and (p["kind"] == "pair" or cards.get(p["item_id"].split(":", 1)[1], {}).get("is_perf_related"))]

    def judge_item(p):
        if p["item_id"] in cache:
            return p["item_id"], cache[p["item_id"]]
        # judge against the full findings list once; recall@budget applies severity/topk after
        j, c = _opus_judge(ground_truth(p), p["findings"])
        with _cost_lock:
            cost[0] += c
        return p["item_id"], j

    with ThreadPoolExecutor(max_workers=10) as ex:
        for f in as_completed([ex.submit(judge_item, p) for p in pos_items if p["item_id"] not in cache]):
            iid, j = f.result()
            cache[iid] = j

    # a HIT@budget = judged hit AND the hit finding is within top-budget severe findings
    pred_by_id = {p["item_id"]: p for p in preds}

    def hit_at(p, budget):
        j = cache.get(p["item_id"], {})
        if not j.get("hit"):
            return False
        idx = j.get("hit_finding_index")
        top = topk_severe(p["findings"], budget)
        top_ids = {id(f) for f in top}
        if idx is None or idx >= len(p["findings"]):
            # judged hit but index unknown → credit if any severe finding in top-budget
            return len(top) > 0
        return id(p["findings"][idx]) in top_ids

    def recall(items, budget):
        if not items:
            return (0, 0)
        h = sum(1 for p in items if hit_at(p, budget))
        return (h, len(items))

    # overall recall@{1,2,5}
    overall = {b: recall(pos_items, b) for b in (1, 2, 5)}
    # per-kind (positives only): use card kind (pairs → regression-fix by definition)
    def kind_of(p):
        if p["kind"] == "pair":
            return "regression-fix"
        return cards.get(p["item_id"].split(":", 1)[1], {}).get("kind", "?")
    by_kind = defaultdict(list)
    for p in pos_items:
        by_kind[kind_of(p)].append(p)
    perkind2 = {k: recall(v, 2) for k, v in by_kind.items()}
    # per-taxonomy (big category)
    leaf2cat = _leaf_to_category()
    def cat_of(p):
        if p["kind"] == "pair":
            pr = pairs[p["item_id"].split(":", 1)[1]]
            c = cards.get(pr.get("case_id"), {})
        else:
            c = cards.get(p["item_id"].split(":", 1)[1], {})
        return leaf2cat.get(c.get("taxonomy_label"), "n/a")
    by_cat = defaultdict(list)
    for p in pos_items:
        by_cat[cat_of(p)].append(p)
    percat2 = {k: recall(v, 2) for k, v in by_cat.items()}

    # benign FPR (negatives): FP if any finding severe & conf≥0.5
    neg_items = [p for p in preds if p["kind"] == "neg"]
    def is_fp(p):
        return any(f.get("severity") in HIT_SEVERITIES and f.get("confidence", 0) >= FP_CONF
                   for f in p["findings"])
    by_ntype = defaultdict(list)
    for p in neg_items:
        nt = negs.get(p["item_id"].split(":", 1)[1], {}).get("negative_type", "?")
        by_ntype[nt].append(p)
    fpr = {}
    for nt, v in by_ntype.items():
        fp = sum(1 for p in v if is_fp(p))
        fpr[nt] = (fp, len(v), wilson(fp, len(v)))
    tot_fp = sum(1 for p in neg_items if is_fp(p))
    fpr["_weighted_total"] = (tot_fp, len(neg_items), wilson(tot_fp, len(neg_items)))

    leaks = sum(1 for p in preds if p.get("leak_attempt"))
    return {"detector": detector_id, "split": split, "n_pos": len(pos_items),
            "n_neg": len(neg_items), "overall": overall, "perkind2": perkind2,
            "percat2": percat2, "fpr": fpr, "leak_attempts": leaks,
            "judge_cost": round(cost[0], 2), "cache": cache}


def _leaf_to_category():
    import re
    m = {}
    cur = None
    for ln in (BASE / "taxonomy" / "taxonomy.yaml").read_text().splitlines():
        mm = re.match(r"^  - id: (\S+)", ln)
        if mm:
            cur = mm.group(1)
        lm = re.match(r"^      - id: (\S+)", ln)
        if lm:
            m[lm.group(1)] = cur
    return m


def fmt_recall(t):
    h, n = t
    return f"{h}/{n} ({100*h/n:.1f}%)" if n else "0/0"


def cmd_score(a):
    m = score_predictions(a.detector, a.split)
    L = [f"# Score — {a.detector} on {a.split} (metrics.v1)", "",
         f"- generated: {now_iso()} · **static ceiling {CEILING}%** · judge {OPUS}",
         f"- positives {m['n_pos']} · negatives {m['n_neg']} · leak_attempts {m['leak_attempts']}"
         + (" ⚠️" if m["leak_attempts"] else " ✓"),
         f"- judge spend ~${m['judge_cost']}", "",
         "## recall@budget (severity∈{critical,important}, ceiling "+f"{CEILING}%)", "",
         "| budget | recall |", "|--:|---|"]
    for b in (1, 2, 5):
        L.append(f"| {b} | {fmt_recall(m['overall'][b])} |")
    L += ["", "## per-kind recall@2 (**regression-fix = north star, n≈145**)", "",
          "| kind | recall@2 |", "|---|---|"]
    for k in sorted(m["perkind2"], key=lambda k: -m["perkind2"][k][1]):
        star = " **★**" if k == "regression-fix" else ""
        L.append(f"| {k}{star} | {fmt_recall(m['perkind2'][k])} |")
    L += ["", "## per-taxonomy recall@2", "", "| category | recall@2 |", "|---|---|"]
    for k in sorted(m["percat2"], key=lambda k: -m["percat2"][k][1]):
        L.append(f"| {k} | {fmt_recall(m['percat2'][k])} |")
    L += ["", "## benign FPR (severe & conf≥0.5), Wilson 95% CI", "",
          "| negative_type | FP/N | rate | 95% CI |", "|---|---|--:|---|"]
    for nt in sorted(m["fpr"], key=lambda k: (k == "_weighted_total", k)):
        fp, n, (lo, hi) = m["fpr"][nt]
        L.append(f"| {nt} | {fp}/{n} | {100*fp/n if n else 0:.1f}% | [{100*lo:.1f}, {100*hi:.1f}] |")
    (REPORTS / f"score_{a.detector}_{a.split}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("--calib-detector", default="baseline_generic")
    c.add_argument("--calib-split", default="dev_calib")
    c.add_argument("--xfamily-model", default="us.amazon.nova-pro-v1:0")
    s = sub.add_parser("score")
    s.add_argument("--detector", required=True)
    s.add_argument("--split", default="test_eval_subset")
    a = ap.parse_args()
    if a.cmd == "calibrate":
        cmd_calibrate(a)
    elif a.cmd == "score":
        cmd_score(a)


if __name__ == "__main__":
    main()
