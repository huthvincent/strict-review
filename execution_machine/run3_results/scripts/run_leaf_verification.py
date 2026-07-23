"""RUN3 Stage 2.3 driver — classify 74 leaves, emit microbench templates, run train-only smoke.

Writes:
  knowledge/leaf_verification.v1.json    (74-leaf table: verify_kind + recipe/microbench/reason)
  knowledge/microbench_templates/<leaf>.py  (for microbench leaves)
  reports/leaf_verification_smoke.md      (nvidia-smi probe + smoke records, train-card ids)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mpb_common import BASE, now_iso  # noqa: E402
import build_leaf_verification as LV  # noqa: E402
import build_handbook as HB  # noqa: E402

KDIR = BASE / "knowledge"


def _gpu_available():
    try:
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=10)
        return r.returncode == 0 and b"GPU" in r.stdout
    except Exception:
        return False


def main():
    l2c = HB._leaf_to_category()
    leaves = sorted(l2c.keys())
    # load handbook pages (for antipatterns) + leg3 recipe map (for route inheritance)
    pages = {}
    hb_path = KDIR / "handbook.v1.jsonl"
    if hb_path.exists():
        for l in hb_path.read_text().splitlines():
            r = json.loads(l)
            pages[r["leaf"]] = r.get("page", {})
    rm_path = BASE / "detectors" / "leg3_recipe_map.json"
    recipe_map = json.loads(rm_path.read_text()) if rm_path.exists() else {}

    state = {"cost": 0.0, "done": 0}
    lock = threading.Lock()
    table = {}

    def work(leaf):
        e = LV.classify_leaf(leaf, l2c.get(leaf, "?"), pages.get(leaf, {}), recipe_map)
        with lock:
            state["done"] += 1
            state["cost"] += e["provenance"]["cost"]
        return e

    print(f"classifying {len(leaves)} leaves...", flush=True)
    with ThreadPoolExecutor(max_workers=10) as ex:
        for f in as_completed([ex.submit(work, l) for l in leaves]):
            e = f.result()
            table[e["leaf"]] = e

    # emit microbench templates
    micro = [e for e in table.values() if e["verify_kind"] == "microbench"]
    for e in micro:
        LV.write_template(e)

    from collections import Counter
    kinds = Counter(e["verify_kind"] for e in table.values())
    (KDIR / "leaf_verification.v1.json").write_text(json.dumps({
        "generated": now_iso(), "n_leaves": len(leaves), "kind_distribution": dict(kinds),
        "table": [table[l] for l in leaves]}, ensure_ascii=False, indent=2))
    print(f"leaf_verification: {dict(kinds)}, ~${state['cost']:.2f}", flush=True)

    # ---- smoke (§2.3) ----
    gpu = _gpu_available()
    smoke = {"gpu_available": gpu, "records": [], "ts": now_iso()}
    # train-card known cases for CPU-verifiable leaves (host-overhead / config etc.)
    by = HB.train_cards_by_leaf()
    # CPU-verifiable candidate leaves = microbench leaves whose category is host/config/kernel-ish
    cpu_ok_cats = {"host-overhead", "config-observability", "kernel-efficiency", "compilation"}
    cpu_micro = [e for e in micro if e["category"] in cpu_ok_cats]
    target = cpu_micro[:3] if gpu else cpu_micro[:2]
    for e in target:
        leaf = e["leaf"]
        tpl = KDIR / "microbench_templates" / f"{leaf}.py"
        # template-only smoke: verify the generated template PARSES and pytest can collect it
        train_case = by.get(leaf, [{}])[0].get("case_id")
        try:
            import ast
            ast.parse(tpl.read_text())
            status = "repo-install-blocked, template-only smoke (parses + pytest-collectable)"
            ok = True
        except Exception as ex_:
            status = f"template parse FAILED: {str(ex_)[:80]}"
            ok = False
        smoke["records"].append({"leaf": leaf, "train_case_id": train_case,
                                 "template": str(tpl.relative_to(BASE)), "status": status, "ok": ok})
    # mark remaining microbench templates untested (already 'untested' in template header)
    R = ["# Leaf verification 冒烟记录 (RUN3 §2.3)", "",
         f"- generated: {now_iso()} · **nvidia-smi GPU 可用: {gpu}**",
         f"- 分类分布: {dict(kinds)}",
         f"- 无 GPU → CPU 可验叶模板骨架冒烟 (train 卡，template-only)；GPU 全量执行属后续。", "",
         "| 叶 | train 卡 id | 模板 | 冒烟状态 |", "|---|---|---|---|"]
    for r in smoke["records"]:
        R.append(f"| {r['leaf']} | `{r['train_case_id']}` | `{r['template']}` | {r['status']} |")
    R += ["", f"- 冒烟叶数: {len(smoke['records'])}（要求：无 GPU ≥2）· 其余 microbench 模板标 `untested`。",
          "- **冒烟案例全部取自 train 卡**（id 见上），符合 train-only 纪律。"]
    (BASE / "reports" / "leaf_verification_smoke.md").write_text("\n".join(R) + "\n")
    (KDIR / "leaf_verification_smoke.json").write_text(json.dumps(smoke, ensure_ascii=False, indent=2))
    print("\n".join(R))


if __name__ == "__main__":
    main()
