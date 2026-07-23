"""RUN4 v3 §4.2 — GENUINE new prospective window: v2.1 on 22 unseen Megatron commits (>=20).
These commits post-date the dataset AND v2.1's design window → true prospective."""
from __future__ import annotations
import json, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mpb_common import BASE, REPORTS, now_iso
import eval_harness as H
import detector_v2_1 as D
REPO="Megatron-LM"; OUT=BASE/"predictions"/"detector_v2_1_newwindow"/"commits.jsonl"; SEV={"critical","important"}
def main():
    w=json.loads((BASE/"prospective"/"new_window_run4.json").read_text()); commits=w["commits"]
    cfg=json.loads(D.CONFIG_PATH.read_text()); detect=D.make_detect(cfg,[])
    OUT.parent.mkdir(parents=True,exist_ok=True)
    done={json.loads(l)["sha"] for l in OUT.read_text().splitlines()} if OUT.exists() else set()
    todo=[c for c in commits if c["sha"] not in done]
    print(f"new-window prospective: {len(commits)} commits, {len(todo)} to run",flush=True)
    lock=threading.Lock(); st={"n":0,"leak":0,"cost":0.0}
    def work(c):
        try:
            v=H.build_view(REPO,c["sha"]); t=H.Tools(REPO,c["sha"],v["parent_sha"])
            t0=time.time(); f,m=detect(v,t)
            return {"sha":c["sha"],"date":c["date"],"subject":c["subject"],"findings":f,"leak_attempt":t.leak_attempt,
                    "gate_predicate":m.get("gate_predicate"),"latency_s":round(time.time()-t0,2),"cost":m.get("cost",0)}
        except Exception as e:
            return {"sha":c["sha"],"date":c["date"],"subject":c["subject"],"findings":[],"error":str(e)[:150],"cost":0}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for f in as_completed([ex.submit(work,c) for c in todo]):
            r=f.result()
            with lock:
                st["n"]+=1; st["cost"]+=r.get("cost",0); st["leak"]+=1 if r.get("leak_attempt") else 0
                with open(OUT,"a") as fh: fh.write(json.dumps(r,ensure_ascii=False)+"\n")
    rows=[json.loads(l) for l in OUT.read_text().splitlines()]
    def severe(r): return [f for f in r["findings"] if f.get("severity") in SEV]
    fire=[r for r in rows if severe(r)]; gated=[r for r in rows if r.get("gate_predicate")]
    L=["# RUN4 v3 §4.2 — 新前瞻窗（22 未见 commit，真前瞻）","",
       f"- generated: {now_iso()} · 冻结 v2.1 · 窗口 {w['window']} · leak={st['leak']}",
       f"- **触发率（severe）: {len(fire)}/{len(rows)}** · 被门谓词命中 {len(gated)}",
       "- 这是数据集与 v2.1 设计窗口**之后**的新提交 → 真前瞻（区别于 §4.1 诊断回放）。","",
       "## 触发 commit（severe，按 conf_raw）",""]
    rows_f=sorted(fire,key=lambda r:-max((f.get("conf_raw",0) for f in severe(r)),default=0))
    for r in rows_f[:12]:
        sv=severe(r)
        L+=[f"### `{r['sha'][:12]}` (conf_raw {max(f.get('conf_raw',0) for f in sv):.2f}) — {r['subject'][:75]}",
            f"- {sv[0].get('severity')}/{sv[0].get('source')}: {(sv[0].get('claim') or '')[:170]}",""]
    open(REPORTS/"prospective_new_v2_1.md","w").write("\n".join(L)+"\n")
    open(REPORTS/"prospective_new_v2_1.json","w").write(json.dumps({"n":len(rows),"fire":len(fire),"gated":len(gated),"leak":st["leak"],"cost":round(st["cost"],2)},indent=2))
    print(f"new-window done: {len(fire)}/{len(rows)} fire, leak={st['leak']}, ${st['cost']:.2f}")
if __name__=="__main__": main()
