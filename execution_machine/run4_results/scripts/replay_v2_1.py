"""RUN4 v3 §4.1 — DIAGNOSTIC replay (NOT prospective) of v2.1 on the same 61 commits.
Charter §4.1: v2.1 recipe design referenced this window's failures (postmortem cited bcf4c8fb),
so this is diagnostic only — MUST NOT be written as a prospective-advantage claim.
"触发" = formal-scoring finding (run4.v3: severe; for triggering we use severe fire).
Dual口径 trigger rate (gate_predicate含/不含). Compare to v1 (severe 24/61) & v2.0 (0/61)."""
from __future__ import annotations
import json, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mpb_common import BASE, REPORTS, now_iso
import eval_harness as H
import detector_v2_1 as D
import usage_ledger as UL
REPO="Megatron-LM"; OUT=BASE/"predictions"/"detector_v2_1_replay"/"commits.jsonl"
SEV={"critical","important"}
def main():
    unseen=json.loads((BASE/"prospective"/"unseen_commits.json").read_text()); commits=unseen["commits"]
    cfg=json.loads(D.CONFIG_PATH.read_text()); detect=D.make_detect(cfg,[])
    OUT.parent.mkdir(parents=True,exist_ok=True)
    done={json.loads(l)["sha"] for l in OUT.read_text().splitlines()} if OUT.exists() else set()
    todo=[c for c in commits if c["sha"] not in done]
    print(f"replay v2.1: {len(commits)} commits, {len(todo)} to run",flush=True)
    lock=threading.Lock(); st={"n":0,"leak":0,"cost":0.0}
    def work(c):
        try:
            view=H.build_view(REPO,c["sha"]); tools=H.Tools(REPO,c["sha"],view["parent_sha"])
            t0=time.time(); f,m=detect(view,tools)
            return {"sha":c["sha"],"date":c["date"],"subject":c["subject"],"findings":f,
                    "leak_attempt":tools.leak_attempt,"gate_predicate":m.get("gate_predicate"),
                    "touches":m.get("touches_perf_surface"),"latency_s":round(time.time()-t0,2),"cost":m.get("cost",0)}
        except Exception as e:
            return {"sha":c["sha"],"date":c["date"],"subject":c["subject"],"findings":[],"error":str(e)[:150],"cost":0}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for f in as_completed([ex.submit(work,c) for c in todo]):
            r=f.result()
            with lock:
                st["n"]+=1; st["cost"]+=r.get("cost",0); st["leak"]+=1 if r.get("leak_attempt") else 0
                with open(OUT,"a") as fh: fh.write(json.dumps(r,ensure_ascii=False)+"\n")
    print(f"replay done: {st['n']} commits, leak={st['leak']}, ${st['cost']:.2f}",flush=True)
    # compare
    v21=[json.loads(l) for l in OUT.read_text().splitlines()]
    def severe(r): return [f for f in r["findings"] if f.get("severity") in SEV]
    def fired(r): return len(severe(r))>0
    v21_fire={r["sha"] for r in v21 if fired(r)}
    v21_fire_nogate={r["sha"] for r in v21 if fired(r) and not r.get("gate_predicate")}
    v21_gated={r["sha"] for r in v21 if r.get("gate_predicate")}
    v1p=BASE/"predictions"/"detector_v1_prospective"/"commits.jsonl"
    v1=[json.loads(l) for l in v1p.read_text().splitlines()] if v1p.exists() else []
    v1_fire={r["sha"] for r in v1 if [f for f in r["findings"] if f.get("severity") in SEV]}
    v20p=BASE/"predictions"/"detector_v2_prospective"/"commits.jsonl"
    v20=[json.loads(l) for l in v20p.read_text().splitlines()] if v20p.exists() else []
    v20_fire={r["sha"] for r in v20 if [f for f in r["findings"] if f.get("severity") in SEV]}
    v21map={r["sha"]:r for r in v21}
    inter=v21_fire&v1_fire; only21=v21_fire-v1_fire; only1=v1_fire-v21_fire
    L=["# RUN4 v3 Stage 4 — 诊断性回放 v2.1（**非前瞻**）","",
       f"- generated: {now_iso()} · 冻结 v2.1 · 窗口 {unseen['window']} · leak={st['leak']}",
       "- ⚠️ **v2.1 配方设计曾参考本窗口失败案例（尸检引用过 bcf4c8fb）；本节仅作诊断对照，",
       "  禁止写成前瞻优势主张。** 真前瞻需新窗（见 §4.2）。",
       f"- **触发率（severe 口径）**：含门 **{len(v21_fire)}/61** · 不含门(排除 gate_predicate) {len(v21_fire_nogate)}/61 · 被门谓词命中 {len(v21_gated)}",
       f"- 对照：v1 severe {len(v1_fire)}/61（RUN2 发布口径 26/61 差异因当时含非-severe 计法）· v2.0 {len(v20_fire)}/61",
       f"- 交并差 vs v1：交集 {len(inter)} · v2.1 独有 {len(only21)} · v1 独有 {len(only1)}","",
       "## v2.1 独有触发 top10（severe，标注门影响）",""]
    def top(shas,m,n=10):
        rows=[]
        for s in shas:
            r=m.get(s,{}); sv=severe(r)
            if sv: rows.append((max(f.get("conf_raw",0) for f in sv),s,r,sv))
        rows.sort(reverse=True); return rows[:n]
    for conf,s,r,sv in top(only21,v21map):
        g=" [gate_predicate=true]" if r.get("gate_predicate") else ""
        L+=[f"### `{s[:12]}` (conf_raw {conf:.2f}){g} — {r['subject'][:75]}",
            f"- {sv[0].get('severity')}/{sv[0].get('source')}: {(sv[0].get('claim') or '')[:180]}",
            f"- suggested_benchmark: {sv[0].get('suggested_benchmark','n/a')}",""]
    open(REPORTS/"replay_v2_1.md","w").write("\n".join(L)+"\n")
    open(REPORTS/"replay_v2_1.json","w").write(json.dumps({"v21_fire":len(v21_fire),"v21_fire_nogate":len(v21_fire_nogate),
        "v21_gated":len(v21_gated),"v1_fire":len(v1_fire),"v20_fire":len(v20_fire),"inter":len(inter),
        "only21":len(only21),"only1":len(only1),"leak":st["leak"]},indent=2))
    print("\n".join(L[:8]))
if __name__=="__main__": main()
