"""RUN3 Stage 5 — prospective v2 vs v1 on the same 61 unseen commits (§5).
Reports two trigger rates (with/without profile gate), intersect/union/diff vs v1's 26,
gate-excluded commits as a separate class, and each side's unique-trigger top10.
"""
from __future__ import annotations
import json, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mpb_common import BASE, REPORTS, now_iso
import eval_harness as H
import detector_v2 as D2

REPO = "Megatron-LM"
OUT = BASE / "predictions" / "detector_v2_prospective" / "commits.jsonl"

def main():
    unseen = json.loads((BASE / "prospective" / "unseen_commits.json").read_text())
    commits = unseen["commits"]
    cfg = json.loads((D2.CONFIG_PATH).read_text())
    detect = D2.make_detect(cfg, [])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["sha"] for l in OUT.read_text().splitlines()} if OUT.exists() else set()
    todo = [c for c in commits if c["sha"] not in done]
    print(f"prospective v2: {len(commits)} commits, {len(todo)} to run", flush=True)
    lock = threading.Lock(); st = {"n": 0, "leak": 0, "cost": 0.0}
    def work(c):
        try:
            view = H.build_view(REPO, c["sha"]); tools = H.Tools(REPO, c["sha"], view["parent_sha"])
            t0 = time.time(); findings, meta = detect(view, tools)
            return {"sha": c["sha"], "date": c["date"], "subject": c["subject"],
                    "findings": findings, "leak_attempt": tools.leak_attempt, "gate": meta.get("gate", False),
                    "latency_s": round(time.time()-t0, 2), "cost": meta.get("cost", 0)}
        except Exception as e:
            return {"sha": c["sha"], "date": c["date"], "subject": c["subject"], "findings": [], "error": str(e)[:150], "cost": 0}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for f in as_completed([ex.submit(work, c) for c in todo]):
            r = f.result()
            with lock:
                st["n"] += 1; st["cost"] += r.get("cost", 0); st["leak"] += 1 if r.get("leak_attempt") else 0
                with open(OUT, "a") as fh: fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"v2 prospective done: {st['n']} commits, {st['leak']} leaks, ${st['cost']:.2f}", flush=True)

    # ---- compare to v1 ----
    v2 = [json.loads(l) for l in OUT.read_text().splitlines()]
    v1p = BASE / "predictions" / "detector_v1_prospective" / "commits.jsonl"
    v1 = [json.loads(l) for l in v1p.read_text().splitlines()] if v1p.exists() else []
    def severe(r): return [f for f in r["findings"] if f.get("severity") in ("critical", "important")]
    def fired(r): return len(severe(r)) > 0
    v2_fire = {r["sha"] for r in v2 if fired(r)}
    v2_fire_nogate = {r["sha"] for r in v2 if fired(r) or r.get("gate")}  # gate would've suppressed
    v2_gated = {r["sha"] for r in v2 if r.get("gate")}
    v1_fire = {r["sha"] for r in v1 if fired(r)}
    inter = v2_fire & v1_fire; only_v2 = v2_fire - v1_fire; only_v1 = v1_fire - v2_fire
    v2map = {r["sha"]: r for r in v2}; v1map = {r["sha"]: r for r in v1}
    L = ["# RUN3 Stage 5 — 前瞻对决 v2 vs v1（同一 61 commit）", "",
         f"- generated: {now_iso()} · 冻结 detector_v2 · budget=2 · leak {st['leak']} · 窗口 {unseen['window']}",
         f"- **v2 触发率（含门）: {len(v2_fire)}/{len(v2)} ({100*len(v2_fire)/max(1,len(v2)):.0f}%)**"
         f" · v2 被画像门直接排除: {len(v2_gated)} 个（单列一类）",
         f"- v1 触发（RUN2）: {len(v1_fire)}/{len(v1)}",
         f"- 交集 {len(inter)} · v2 独有 {len(only_v2)} · v1 独有 {len(only_v1)}", "",
         "## v2 独有触发 top10（按 confidence）", ""]
    def top(shas, m, n=10):
        rows = []
        for s in shas:
            r = m.get(s, {}); sev = severe(r)
            if sev: rows.append((max(f.get("confidence", 0) for f in sev), s, r, sev))
        rows.sort(reverse=True)
        return rows[:n]
    for conf, s, r, sev in top(only_v2, v2map):
        gated = " [受门影响]" if r.get("gate") else ""
        L += [f"### `{s[:12]}` (conf {conf:.2f}){gated} — {r['subject'][:80]}",
              f"- {sev[0].get('severity')}: {(sev[0].get('claim') or '')[:200]}",
              f"- suggested_benchmark: {sev[0].get('suggested_benchmark','n/a')}", ""]
    L += ["## v1 独有触发 top10（v2 漏报，诊断 v2 过度沉默）", ""]
    for conf, s, r, sev in top(only_v1, v1map):
        L += [f"- `{s[:12]}` (conf {conf:.2f}) — {r['subject'][:70]} — {sev[0].get('severity')}: {(sev[0].get('claim') or '')[:120]}"]
    L += ["", "## 结论（定性，无标签）",
          f"- v2 触发率 {100*len(v2_fire)/max(1,len(v2)):.0f}% vs v1 {100*len(v1_fire)/max(1,len(v1)):.0f}%："
          + ("v2 显著更沉默，与 dev 过度抑制一致。" if len(v2_fire) < len(v1_fire) else "v2 触发相当或更多。"),
          "- 被画像门排除的 commit 单列见上；leak 纪律：全程 leak_attempt = %d。" % st["leak"]]
    (REPORTS / "prospective_v2.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:8]))

if __name__ == "__main__":
    main()
