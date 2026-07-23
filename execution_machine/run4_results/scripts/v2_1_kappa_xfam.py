"""RUN4 v3 §2.3 — κ recheck (100, pos/neg×HIT/MISS strata, 2nd judge bypasses cache w/ nonce,
not written back) ≥0.7; cross-family Nova 50 (over-sample HIT). run4.v3 recall口径 = severe
finding judged HIT (no conf threshold)."""
from __future__ import annotations
import json, random, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mpb_common import BASE, CASES, PAIRS, read_jsonl, now_iso
import eval_metrics as M
SEV={"critical","important"}; SEED=20260723
def gt(r,cards,pairs):
    if r["kind"]=="pair":
        pr=pairs[r["item_id"].split(":",1)[1]]; c=cards.get(pr.get("case_id"),{})
        return {"symptom":pr.get("symptom"),"mechanism":c.get("mechanism",""),"evidence":c.get("evidence","")}
    c=cards.get(r["item_id"].split(":",1)[1],{})
    return {"symptom":c.get("symptom"),"mechanism":c.get("mechanism",""),"evidence":c.get("evidence","")}
def severe(r): return [f for f in r["findings"] if f.get("severity") in SEV]
def judge_nocache(g,fs,nonce):
    u=M._judge_user(g,fs)+f"\n\n[recheck-nonce:{nonce}]"
    with M._cl().messages.stream(model=M.OPUS,max_tokens=600,thinking={"type":"disabled"},system=M._judge_system(),messages=[{"role":"user","content":u}],tools=[M.JUDGE_EMIT],tool_choice={"type":"tool","name":"emit_judgement"}) as st:
        r=st.get_final_message()
    return next((x.input for x in r.content if x.type=="tool_use"),{}) or {}
def main():
    cards={c["case_id"]:c for c in read_jsonl(CASES/"cards_final_v011.jsonl")}
    pairs={p["pair_id"]:p for p in read_jsonl(PAIRS/"regression_pairs_v011.jsonl")}
    rows=[json.loads(l) for l in (BASE/"predictions"/"detector_v2_1"/"dev_tune.jsonl").read_text().splitlines()]
    rnd=random.Random(SEED)
    # strata by pos/neg × HIT/MISS (HIT = severe finding judged hit)
    strata=defaultdict(list)
    for r in rows:
        if r["kind"] in ("case","pair"):
            sv=severe(r)
            hit=False
            if sv:
                j,_=M._opus_judge(gt(r,cards,pairs),sv); hit=bool(j.get("hit"))
            strata[("pos",hit)].append((r,hit))
        elif r["kind"]=="neg":
            fp=any(f.get("severity") in SEV and f.get("conf_raw",0)>=0.5 for f in r["findings"])
            strata[("neg",fp)].append((r,fp))
    ksample=[]
    for key,items in strata.items():
        ksample+=rnd.sample(items,min(25,len(items)))
    ksample=ksample[:100]
    a1,a2=[],[]
    for i,(r,_) in enumerate(ksample):
        sv=severe(r) or r["findings"]
        j1,_=M._opus_judge(gt(r,cards,pairs),sv)
        j2=judge_nocache(gt(r,cards,pairs),sv,f"n{i}")
        a1.append(str(bool(j1.get("hit")))); a2.append(str(bool(j2.get("hit"))))
    kappa=M.cohens_kappa(a1,a2); agree=sum(1 for x,y in zip(a1,a2) if x==y)/len(a1) if a1 else 0
    # cross-family Nova 50, over-sample HIT (pos HIT)
    hits=[r for (r,h) in strata[("pos",True)]]
    others=[r for (r,_) in strata[("pos",False)]+strata[("neg",False)]+strata[("neg",True)]]
    xf_items=(rnd.sample(hits,min(30,len(hits)))+rnd.sample(others,min(20,len(others))))[:50]
    gap_note = "" if len(hits)>=30 else f"（HIT 仅 {len(hits)}，过采到全部+补齐；缺口因 recall 本身）"
    xf=[]
    for r in xf_items:
        g=gt(r,cards,pairs); sv=severe(r) or r["findings"]
        jo,_=M._opus_judge(g,sv)
        try: jx=M._nonanthropic_judge(g,sv,"us.amazon.nova-pro-v1:0")
        except Exception: continue
        xf.append((bool(jo.get("hit")),bool(jx.get("hit"))))
    xfa=sum(1 for x,y in xf if x==y)/len(xf) if xf else 0
    L=["# detector_v2.1 κ 复检 + 跨家族 (RUN4 v3 §2.3)","",
       f"- generated: {now_iso()} · judge {M.OPUS}",
       f"- **κ 复检（{len(a1)} 项，绕缓存加 nonce 不写回）：一致率 {agree:.1%} · κ={kappa:.3f}** "+("✓≥0.7" if (kappa or 0)>=0.7 else "✗<0.7 停机"),
       f"- **跨家族 Nova（{len(xf)} 项，过采 HIT）：与 Opus-judge 一致率 {xfa:.1%}**{gap_note}"]
    open(BASE/"reports"/"v2_1_kappa_xfam.md","w").write("\n".join(L)+"\n")
    open(BASE/"reports"/"v2_1_kappa_xfam.json","w").write(json.dumps({"kappa":kappa,"agree":agree,"xfam_agree":xfa,"n_k":len(a1),"n_xf":len(xf)},indent=2))
    print("\n".join(L))
if __name__=="__main__": main()
