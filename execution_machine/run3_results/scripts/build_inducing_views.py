"""RUN3 Stage 1.3 — build inducing-side PR-time views for dev regfix cases (§1.3).

For dev regfix cases with inducing_commit_traceable ∈ {direct, likely} AND the inducing
sha resolvable in the local clone → build the INDUCING-side PR-time view (view at the
inducing commit, leak-safe). Writes splits/dev_regfix_inducing_views.jsonl and reports
conversion rate + converted-subset composition (n, detectability, repo).

Also computes diff-size distribution for framing §1.2 → reports/framing_diffsize.json.

The inducing sha comes from the card's `inducing_ref`/pair linkage. We resolve it via the
pairs file (case_id → inducing_sha) first, else parse inducing_ref for a sha-like token.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mpb_common import BASE, CASES, PAIRS, SPLITS, REPORTS, read_jsonl, now_iso  # noqa: E402
import eval_harness as H  # noqa: E402

SHA_RE = re.compile(r"\b([0-9a-f]{7,40})\b")


def _resolve_inducing(card, pairs_by_case):
    """Return inducing sha if we can find one, else None."""
    # 1) via pairs file: a pair whose case_id == this card's case_id gives inducing_sha
    pr = pairs_by_case.get(card["case_id"])
    if pr and pr.get("inducing_sha"):
        return pr["inducing_sha"]
    # 2) parse inducing_ref for a sha-like token
    ref = str(card.get("inducing_ref") or "")
    m = SHA_RE.search(ref)
    if m:
        return m.group(1)
    return None


def main():
    cards = {c["case_id"]: c for c in read_jsonl(CASES / "cards_final_v011.jsonl")}
    pairs = list(read_jsonl(PAIRS / "regression_pairs_v011.jsonl"))
    pairs_by_case = {p["case_id"]: p for p in pairs if p.get("case_id")}

    dev = [l.strip() for l in (SPLITS / "dev.txt").read_text().splitlines() if l.strip()]
    dev_regfix = [cards[l.split(":", 1)[1]] for l in dev if l.startswith("case:")
                  and cards.get(l.split(":", 1)[1], {}).get("is_perf_related")
                  and cards.get(l.split(":", 1)[1], {}).get("kind") == "regression-fix"]

    traceable = [c for c in dev_regfix if c.get("inducing_commit_traceable") in ("direct", "likely")]

    views = []
    fail_resolve = 0
    fail_git = 0
    diff_sizes = {"case": [], "inducing": []}
    for c in traceable:
        ind_sha = _resolve_inducing(c, pairs_by_case)
        if not ind_sha:
            fail_resolve += 1
            continue
        try:
            view = H.build_view(c["repo"], ind_sha)  # inducing-side PR-time view (leak-safe)
        except Exception:
            fail_git += 1
            continue
        views.append({
            "case_id": c["case_id"], "repo": c["repo"],
            "fix_sha": c["sha"], "inducing_sha": ind_sha,
            "static_detectability": c.get("static_detectability"),
            "traceable": c.get("inducing_commit_traceable"),
            "view": view,
        })
        diff_sizes["inducing"].append(len(view.get("diff", "")))
        # also record the fix-side diff size for §1.2
        try:
            fv = H.build_view(c["repo"], c["sha"])
            diff_sizes["case"].append(len(fv.get("diff", "")))
        except Exception:
            pass

    (SPLITS / "dev_regfix_inducing_views.jsonl").write_text(
        "".join(json.dumps(v, ensure_ascii=False) + "\n" for v in views))

    conv_rate = len(views) / len(traceable) if traceable else 0
    comp_det = dict(Counter(v["static_detectability"] for v in views))
    comp_repo = dict(Counter(v["repo"] for v in views))

    def stats(xs):
        if not xs:
            return {"n": 0}
        xs = sorted(xs)
        return {"n": len(xs), "median": xs[len(xs) // 2], "mean": round(sum(xs) / len(xs))}
    (REPORTS / "framing_diffsize.json").write_text(json.dumps({
        "fix_case_diffchars": stats(diff_sizes["case"]),
        "inducing_diffchars": stats(diff_sizes["inducing"]),
        "note": "diff 大小（字符）分布，对比 fix-case 视图 vs inducing 视图（converted 子集）"},
        ensure_ascii=False, indent=2))

    summary = {
        "dev_regfix_total": len(dev_regfix),
        "traceable_direct_or_likely": len(traceable),
        "converted": len(views),
        "conversion_rate_of_traceable": round(conv_rate, 3),
        "fail_resolve_sha": fail_resolve, "fail_git_resolve": fail_git,
        "converted_composition": {"detectability": comp_det, "repo": comp_repo},
    }
    (REPORTS / "inducing_conversion.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nwrote splits/dev_regfix_inducing_views.jsonl ({len(views)} views)")


if __name__ == "__main__":
    main()
