"""Stage 6 — prospective run on unseen commits (RUN2_INSTRUCTIONS.md §6).

Runs the FROZEN detector_v1 on Megatron-LM commits the dataset never saw
(prospective/unseen_commits.json, git-fetched post-2026-06-15). No labels →
qualitative report reports/prospective_run.md:
  - per finding block: commit, finding text, confidence, triggering leg, suggested_benchmark
  - sorted by confidence desc; top-20 as a section for human review
  - stats: trigger rate, leg distribution, taxonomy distribution

Same harness (leak-free PR-time view), same frozen config, budget=2.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mpb_common import BASE, REPORTS, now_iso  # noqa: E402
import eval_harness as H  # noqa: E402
import detector_v1 as D  # noqa: E402

REPO = "Megatron-LM"
OUT = BASE / "predictions" / "detector_v1_prospective" / "commits.jsonl"


def main():
    unseen = json.loads((BASE / "prospective" / "unseen_commits.json").read_text())
    commits = unseen["commits"]
    cfg = json.loads((BASE / "detectors" / "detector_v1.config.json").read_text())
    detect = D.make_detect(cfg, [])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["sha"] for l in OUT.read_text().splitlines()} if OUT.exists() else set()
    todo = [c for c in commits if c["sha"] not in done]
    print(f"prospective: {len(commits)} unseen commits ({unseen['window']}), {len(todo)} to run", flush=True)

    lock = threading.Lock()
    st = {"n": 0, "leak": 0, "cost": 0.0}

    def work(c):
        sha = c["sha"]
        try:
            view = H.build_view(REPO, sha)
            tools = H.Tools(REPO, sha, view["parent_sha"])
            t0 = time.time()
            findings, meta = detect(view, tools)
            return {"sha": sha, "date": c["date"], "subject": c["subject"],
                    "findings": findings, "leak_attempt": tools.leak_attempt,
                    "latency_s": round(time.time() - t0, 2),
                    "legs": meta.get("legs"), "routed_leaf": meta.get("routed_leaf"),
                    "cost": meta.get("cost", 0)}
        except Exception as e:
            return {"sha": sha, "date": c["date"], "subject": c["subject"],
                    "findings": [], "error": str(e)[:150], "cost": 0}

    with ThreadPoolExecutor(max_workers=8) as ex:
        for fut in as_completed([ex.submit(work, c) for c in todo]):
            r = fut.result()
            with lock:
                st["n"] += 1
                st["cost"] += r.get("cost", 0)
                st["leak"] += 1 if r.get("leak_attempt") else 0
                with open(OUT, "a") as fh:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"prospective done: {st['n']} commits, {st['leak']} leaks, ${st['cost']:.2f}", flush=True)

    # ---- qualitative report ----
    rows = [json.loads(l) for l in OUT.read_text().splitlines()]
    flagged = [r for r in rows if r["findings"]]
    all_findings = []
    for r in rows:
        for f in r["findings"]:
            all_findings.append((f.get("confidence", 0), r, f))
    all_findings.sort(key=lambda x: -x[0])

    from collections import Counter
    leg_dist = Counter()
    tax_dist = Counter()
    for _, r, f in all_findings:
        cat = f.get("category", "")
        leg = "leg1" if cat.startswith("static-rule") else ("leg3" if cat.startswith("risk-route") else "leg2")
        leg_dist[leg] += 1
        if r.get("routed_leaf"):
            tax_dist[r["routed_leaf"]] += 1

    L = ["# Stage 6 — 真实新 commit 前瞻试运行 (§6)", "",
         f"- generated: {now_iso()} · 冻结 detector_v1 · budget=2 · leak_attempts {st['leak']}",
         f"- 窗口 {unseen['window']}：{len(rows)} 个**数据集未见**的 Megatron-LM 新 commit"
         f"（{unseen.get('note','')}）",
         f"- **触发率**：{len(flagged)}/{len(rows)} commit 被报了问题（{100*len(flagged)/len(rows) if rows else 0:.0f}%）",
         f"- 腿分布：{dict(leg_dist)} · 路由 taxonomy 分布：{dict(tax_dist.most_common(8))}",
         "", "> 无标签 → 定性产出。top-20（按 confidence 降序）供 Rui 人工核验。", "",
         "## Top-20 finding（人工核验清单）", ""]
    for i, (conf, r, f) in enumerate(all_findings[:20], 1):
        leg = f.get("category", "")
        L += [f"### {i}. `{r['sha'][:12]}` (conf {conf:.2f}) — {r['date'][:10]}",
              f"- commit: {r['subject'][:100]}",
              f"- finding [{f.get('severity')}]: {f.get('claim','')[:300]}",
              f"- category/leg: {leg}"
              + (f" · suggested_benchmark: {f.get('suggested_benchmark')}" if f.get('suggested_benchmark') else ""),
              "- 人工核验: `[ ] 真问题  [ ] 误报  [ ] 需查证`", ""]

    L += ["## 全部触发 commit（简表）", "", "| sha | date | #findings | top severity | subject |", "|---|---|--:|---|---|"]
    for r in sorted(flagged, key=lambda r: r["date"], reverse=True):
        sev = max((f.get("severity", "suggestion") for f in r["findings"]),
                  key=lambda s: {"critical": 3, "important": 2, "suggestion": 1}.get(s, 0))
        L.append(f"| {r['sha'][:12]} | {r['date'][:10]} | {len(r['findings'])} | {sev} | {r['subject'][:60]} |")

    L += ["", "## 与 claude[bot] 实际评论对比",
          "- clone 的 PR 元数据不含 review 评论正文，无法离线取得 claude[bot] 评论 → **跳过此对比并记录**（§6）。",
          "", "## 价值说明",
          "- 若 top-20 中有真问题，即『这套东西能用』的最直接证据，也是给 NVIDIA 的第一份材料。",
          "- 本节为定性；未做 GPU 复现，finding 真伪待人工/CI 核验。"]
    (REPORTS / "prospective_run.md").write_text("\n".join(L) + "\n")
    print(f"wrote reports/prospective_run.md ({len(flagged)}/{len(rows)} flagged)", flush=True)


if __name__ == "__main__":
    main()
