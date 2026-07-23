"""RUN3 Stage 2.1② — leak audit for the three train-only assets (handbook / repo_profile /
leaf_verification). Regex-scan for every sha / issue# / PR# / date and cross-check each
against the TRAIN card set; any external reference (not attributable to a train card) is
DELETED from the asset and logged → reports/handbook_leak_audit.md.

Allowed identifiers = shas/issue/PR numbers/dates that appear in the train perf cards'
own fields (sha, inducing_ref, issue_refs, pr_ref, subject, date, mechanism, evidence).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mpb_common import BASE, CASES, SPLITS, read_jsonl, now_iso  # noqa: E402

KDIR = BASE / "knowledge"

SHA_RE = re.compile(r"\b[0-9a-f]{7,40}\b")
NUM_RE = re.compile(r"#(\d{2,6})\b")            # issue/PR style
DATE_RE = re.compile(r"\b20\d{2}[-/]\d{1,2}[-/]\d{1,2}\b")


def _allowed_tokens():
    """All sha/number/date-like tokens that legitimately come from TRAIN cards."""
    cards = {c["case_id"]: c for c in read_jsonl(CASES / "cards_final_v011.jsonl")}
    train = set(l.strip().split(":", 1)[1] for l in (SPLITS / "train.txt").read_text().splitlines()
                if l.strip().startswith("case:"))
    shas, nums, dates = set(), set(), set()
    for cid, c in cards.items():
        if cid not in train:
            continue
        blob = " ".join(str(c.get(k) or "") for k in
                        ("sha", "inducing_ref", "issue_refs", "pr_ref", "subject", "date", "mechanism", "evidence"))
        for m in SHA_RE.findall(blob):
            shas.add(m); shas.add(m[:12]); shas.add(m[:7])
        # numbers: from issue_refs/pr_ref/subject (#NNN)
        for m in NUM_RE.findall(blob):
            nums.add(m)
        for m in re.findall(r"\b\d{2,6}\b", str(c.get("issue_refs") or "") + " " + str(c.get("pr_ref") or "")):
            nums.add(m)
        for m in DATE_RE.findall(blob):
            dates.add(m)
    return shas, nums, dates


def audit_text(text, shas, nums, dates):
    """Return (violations, cleaned_text). Violations = external identifiers found."""
    viol = []
    # sha-like tokens: flag if not a prefix/suffix of any allowed sha
    for tok in set(SHA_RE.findall(text)):
        # skip pure-decimal (handled as numbers) and very common words already hex? sha needs a-f digit mix
        if tok.isdigit():
            continue
        if not any(tok == s or s.startswith(tok) or tok.startswith(s) for s in shas):
            viol.append(("sha", tok))
    for tok in set(NUM_RE.findall(text)):
        if tok not in nums:
            viol.append(("issue/PR#", "#" + tok))
    for tok in set(DATE_RE.findall(text)):
        if tok not in dates:
            viol.append(("date", tok))
    cleaned = text
    for _, tok in viol:
        cleaned = cleaned.replace(tok, "[REDACTED-EXTERNAL]")
    return viol, cleaned


def main():
    shas, nums, dates = _allowed_tokens()
    print(f"allowed train tokens: {len(shas)} sha, {len(nums)} nums, {len(dates)} dates", flush=True)
    assets = {
        "handbook.v1.md": KDIR / "handbook.v1.md",
        "repo_profile.v1.json": KDIR / "repo_profile.v1.json",
        "leaf_verification.v1.json": KDIR / "leaf_verification.v1.json",
    }
    report = ["# 三资产泄漏审计 (RUN3 §2.1②)", "",
              f"- generated: {now_iso()} · 扫描 sha/issue#/PR#/日期，跨对 train 卡集合，外源引用即删",
              f"- 允许 token 池：{len(shas)} sha · {len(nums)} issue/PR# · {len(dates)} 日期（均来自 train 卡）", ""]
    total_viol = 0
    for name, path in assets.items():
        if not path.exists():
            report.append(f"## {name} — ⚠️ 文件不存在，跳过"); continue
        text = path.read_text()
        viol, cleaned = audit_text(text, shas, nums, dates)
        total_viol += len(viol)
        report += [f"## {name}", f"- 外源引用: **{len(viol)}**"]
        if viol:
            for kind, tok in viol[:40]:
                report.append(f"  - [{kind}] `{tok}` → 已删除")
            path.write_text(cleaned)  # scrub in place
            report.append(f"- 已就地删除并留痕（{len(viol)} 处 → `[REDACTED-EXTERNAL]`）")
        else:
            report.append("- ✓ 无外源引用，全部标识符可溯源到 train 卡")
        report.append("")
    report += [f"## 汇总", f"- 三资产外源引用合计: **{total_viol}**"
               + ("（已全部删除）" if total_viol else "（干净）")]
    (BASE / "reports" / "handbook_leak_audit.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
