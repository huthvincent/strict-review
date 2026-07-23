"""RUN4 v3 §3 — ablation report: v2.1 (full) vs v2.1 − route independent leg (80-item subset).
run4.v3 scoring: recall = severe judged HIT (no conf threshold); FP = severe∧conf_raw≥0.5.
Wilson 95% CI; CI overlap → 不可区分."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mpb_common import BASE, CASES, PAIRS, read_jsonl, now_iso
import eval_metrics as M
SEV={"critical","important"}
def gt(r,cards,pairs):
    if r["kind"]=="pair":
        pr=pairs[r["item_id"].split(":",1)[1]]; c=cards.get(pr.get("case_id"),{})
        return {"symptom":pr.get("symptom"),"mechanism":c.get("mechanism",""),"evidence":c.get("evidence","")}
    c=cards.get(r["item_id"].split(":",1)[1],{})
    return {"symptom":c.get("symptom"),"mechanism":c.get("mechanism",""),"evidence":c.get("evidence","")}
def score(det, split, cards, pairs, restrict=None):
    p=BASE/"predictions"/det/f"{split}.jsonl"
    rows=[json.loads(l) for l in p.read_text().splitlines()]
    if restrict is not None: rows=[r for r in rows if r["item_id"] in restrict]
    pos=[r for r in rows if r["kind"] in ("case","pair")]
    neg=[r for r in rows if r["kind"]=="neg"]
    h=0
    for r in pos:
        sv=[f for f in r["findings"] if f.get("severity") in SEV]
        if not sv: continue
        j,_=M._opus_judge(gt(r,cards,pairs),sv)
        if j.get("hit"): h+=1
    fp=sum(1 for r in neg if any(f.get("severity") in SEV and f.get("conf_raw",0)>=0.5 for f in r["findings"]))
    return {"recall":(h,len(pos)),"fpr":(fp,len(neg))}
def ci(t): k,n=t; lo,hi=M.wilson(k,n); return (100*k/n if n else 0,100*lo,100*hi)
def main():
    cards={c["case_id"]:c for c in read_jsonl(CASES/"cards_final_v011.jsonl")}
    pairs={p["pair_id"]:p for p in read_jsonl(PAIRS/"regression_pairs_v011.jsonl")}
    sub=set(l.strip() for l in (BASE/"splits"/"v2_ablation_subset.txt").read_text().splitlines() if l.strip())
    # full v2.1 restricted to the ablation subset (reuse dev_tune preds)
    full=score("detector_v2_1","dev_tune",cards,pairs,restrict=sub)
    abl=score("v2_1_ablate_route","v2_ablation_subset",cards,pairs)
    def overlap(a,b):
        _,alo,ahi=ci(a["recall"]); _,blo,bhi=ci(b["recall"]); return not (ahi<blo or bhi<alo)
    L=["# detector_v2.1 消融 — v2.1 − route 独立腿 (RUN4 v3 §3)","",
       f"- generated: {now_iso()} · 80 项子集 · run4.v3 口径 · Wilson 95% CI","",
       "| 变体 | recall | 95% CI | benign FPR |","|---|---|---|---|"]
    for name,m in [("v2.1 (full)",full),("v2.1 − route 独立腿 (=RUN3 错误形态)",abl)]:
        r,lo,hi=ci(m["recall"]); fp=m["fpr"]
        L.append(f"| {name} | {m['recall'][0]}/{m['recall'][1]} ({r:.1f}%) | [{lo:.1f},{hi:.1f}] | {fp[0]}/{fp[1]} ({100*fp[0]/fp[1] if fp[1] else 0:.1f}%) |")
    L+=["",f"- v2.1 vs −route: {'CI 重叠 → 不可区分' if overlap(full,abl) else 'route 独立腿有可区分贡献'}",
        "- 消融含义：去掉 route 独立腿 = RUN3 v2.0 的错误形态（分诊只挑叶不产 finding）。"]
    open(BASE/"reports"/"v2_1_ablation.md","w").write("\n".join(L)+"\n")
    open(BASE/"reports"/"v2_1_ablation.json","w").write(json.dumps({"full":full,"ablate_route":abl},indent=2))
    print("\n".join(L))
if __name__=="__main__": main()
