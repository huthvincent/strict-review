# Phase 0 — Blockers & Decisions Needed from Rui

> Live status of what stops the paid pipeline from proceeding. Updated as I go.
> Everything **not** listed here is unblocked and either done or in progress
> (see `reports/EXECUTION_STATUS.md`).

---

## 🔴 ACTIVE BLOCKER — Fable 5 unusable on this Bedrock account (blocks §6 arbitration)

**Decision.** Rui chose (2026-07-16) to run all §6 arbitration on **Fable 5**.
The pipeline is built and ready: `scripts/arbitrate_fable.py` + prompt
`prompts/arbitration.v1.md`; full methodology in `reports/ARBITRATION.md`.

**Blocker.** Every Fable 5 request on this account returns:
`400 - data retention mode 'default' is not available for this model`
(reproduced on both `us.` and `global.anthropic.claude-fable-5`, direct
`AnthropicBedrock` and via `shared.llm`). Fable requires **30-day data
retention**; this account is on `default` retention — an org/account-level
Bedrock setting that cannot be changed from the instance.

**To unblock (needs Rui / account admin), any one of:**
- Set this Bedrock account/workspace's data-retention mode to 30-day (Fable's
  requirement), or
- Provide a Fable-capable credential / region / account, or
- Fall back to **Opus 4.8 as the arbiter** (already works here; ~$73 for all
  three queues; still machine-arbitration, just a weaker/ same-family model than
  Fable — a smaller strong-model gap over the Opus-4.8 annotators).

Until then arbitration has NOT run; `cases/audits/audit_log.jsonl` has no
fable-5 records, and `apply_audits.py` produces `cards_final.jsonl` == machine
labels (0 overrides). The split cannot be frozen as "arbitrated" yet.

---

## 🔴 BLOCKER 1 — Pilot gold data is not on this machine (gates the Tier-1 calibration gate)

**What the plan requires.** intro.md §3 (Tier 1) makes the calibration gate a
hard precondition for the full batch: *"用 pilot 的 48 张 gold cards + 1,226 关键词
候选跑校准批… 调阈值使对已知阳性 recall ≥ 95%… 不通过不放全量。"* The §8.8 DoD for
"Tier 1 校准" is literally: *"对 pilot 已知阳性 recall ≥95%；报告落盘"*.

**The problem.** The plan says the pilot artifacts live at
`~/Desktop/OPB/perf-agent/phase0/pilot/` — a **macOS Desktop path**. This is a
headless Linux EC2 box (`/home/ec2-user`). That path does not exist here, and a
scoped scan of `/mnt/efs/tsfm/rui` found **no pilot gold cards, no 1,226-keyword
candidate list, and no 48-sample labels** anywhere on the shared drive. So I
cannot:

1. Run the calibration gate (no known-positives set → can't measure recall@threshold).
2. Ground the Tier-1 few-shots in real gold cards (§3 wants 8–10 real ones; the
   shipped `tier1_screen.v1` prompt uses **hand-written synthetic exemplars** as
   a placeholder — flagged in-file).

**Impact / what's gated.** The full Tier-1 batch (~$629, the spend that unlocks
everything downstream), and transitively Tier 2 / Tier 3 / negatives / splits.
I have **not** spent batch money — only ~$0.90 on live smokes to prove the
pipeline.

**Options for Rui (pick one):**

- **(A) Provide the pilot artifacts.** Copy the pilot dir onto this box (or to
  `/mnt/efs/tsfm/rui/GAA/ai_infra/raw/pilot/`). Fastest path to a faithful
  calibration gate. I'll wire it in and bump the prompt to `tier1_screen.v2`
  with real few-shots.
- **(B) Authorize a bootstrap calibration set built here (~$40–80, recommended
  if the pilot is gone).** I hand-label a stratified ~150–250 commit sample
  (keyword-hit + random) with Opus as a first pass, you spot-audit the positives,
  and we use that as the recall yardstick. Weaker than the real pilot but
  unblocks the batch with a defensible gate. Documented as a deviation in the
  DATASHEET.
- **(C) Proceed to full batch with NO calibration gate.** Not recommended —
  violates §3/§10 recall-first discipline; a silent Tier-1 miss is unrecoverable.

**My recommendation:** (A) if the pilot data can be found; otherwise (B).

---

## 🟡 MINOR — `gh` CLI not installed, no `GH_TOKEN`

Not needed yet (Tier 0/1 are fully offline against local clones). It **is**
needed for Tier 2 `fetch_pr`, and for S2 (issues) / S3 (reverts). Before those
stages I'll need either the `gh` CLI + a token, or a plain `curl` + PAT against
the GitHub REST API (with the §7.4 on-disk cache in `raw/gh_cache/`). Flagging
now so a token can be provisioned in parallel. No decision needed today.

---

## ✅ Confirmed working (no action needed)

- Opus 4.8 live on Bedrock via `us.anthropic.claude-opus-4-8` (smoke returned OK).
- Bedrock **Batch** API usable on this account (existing completed job; S3 bucket
  `omny-artifacts-us-east-1-beta` + `BedrockBatchInferenceRole` present).
- All 4 repos **fully cloned** (not blobless — offline `git show` works; the §7.1
  landmine is defused). Commit counts match the plan (33,089 vs 33,085).
- Tier 1 economics measured live: **~$629 projected** for the full batch (plan: ~$700).
