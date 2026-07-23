# §6 Arbitration — Methodology & Provenance (LLM arbiter)

- decided by: Rui — two decisions:
  - 2026-07-16a: "人工仲裁部分全部交给 fable" (delegate arbitration to an LLM,
    not do it by hand).
  - 2026-07-16b: after Fable 5 proved **unusable on this Bedrock account**
    (`400 - data retention mode 'default' is not available for this model`; Fable
    requires 30-day retention, account is on `default`), "那就用 opus 4.8 把仲裁
    跑了吧" — **fall back to Opus 4.8 as the arbiter.**
- executed by: `scripts/arbitrate_fable.py` (script name kept; arbiter model is
  overridable via `MPB_ARBITER_MODEL`), prompt `prompts/arbitration.v1.md`
- **arbiter model actually used: `claude-opus-4-8`** (`us.anthropic.claude-opus-4-8`
  on Bedrock, $5/$25 per MTok). Fable 5 was the intended arbiter but is blocked on
  this account — see `reports/BLOCKERS.md`. `audit_log.jsonl` records the true
  model per verdict in `provenance.model`, and `auditor: "opus-4-8"`.

## What changed vs the plan

The plan (`intro.md` §6) specifies **human arbitration by Rui** as the dataset's
human ground-truth anchor: the arbitration queue = Tier-2 double-annotation
disagreements + every A/B pair reviewed one-by-one + a 10% stratified card audit,
with Rui's verdicts layered append-only onto the machine labels.

Rui elected to **delegate all arbitration to an LLM** instead of doing it by hand.
The intended arbiter was Fable 5 (a stronger, different-family model than the
Opus-4.8 annotators); because Fable is blocked on this account, the arbiter is
**Opus 4.8**. This document records exactly how the LLM arbiter is used and what
it means for the dataset's claims. **This is the single most important honesty
caveat in v0.1** and is mirrored in `DATASHEET.md`.

### Consequence (state this in any paper/eval)

- v0.1 is **fully machine-labeled AND machine-arbitrated**. There is **no human
  ground truth** in the dataset. The Tier-2 inter-annotator κ=0.81 is
  Opus-vs-Opus agreement, and the arbitration on top is **Opus-vs-Opus** — both
  are machine–machine, and (with the Opus fallback) *same-family*. Do not describe
  any subset as "human-verified" or "gold" in the human sense.
- **The arbiter is the same model family as the annotators.** This is weaker than
  the intended Fable arbitration: the arbiter (Opus 4.8) is the *same model* that
  produced the Tier-1/Tier-2 labels, so it is NOT an independent check and shares
  the annotators' blind spots by construction. It functions as a careful
  second-pass re-read with fresh context and the full diff (blind-first — it forms
  its own judgement before seeing the two labels), which still catches
  self-inconsistencies and diff-vs-label mismatches, but it cannot surface an
  error class the Opus family systematically makes. Treat arbitrated fields as
  *machine-consensus*, not *verified*. Fable would have given a genuine
  stronger-model tie-break; Opus does not.
- If human-anchored gold — or the intended Fable tie-break — is needed later, the
  Opus arbitration does not have to be redone: `audit_log.jsonl` is append-only,
  so a human (or Fable) pass simply appends higher-priority records that supersede
  the Opus verdicts by timestamp (see "Layering" below).

## How Fable is used (mechanics)

**Blind-first, evidence-bound, fully local.** For each queued item the arbiter:

1. Receives the contested record and the two annotators' opinions, but the prompt
   (`arbitration.v1`) instructs it to **form its own judgement from the primary
   evidence first** (the commit diff + git history) and only *then* weigh the
   annotator labels — reducing anchoring on the machine labels it is meant to check.
2. Investigates with the **same local git tools Tier 2 used** — `git_show(sha,
   path?)` and `git_log(args)` (e.g. `-S "symbol"` pickaxe to check whether an
   earlier commit truly introduced a regression). No network; no GitHub.
3. Emits a structured `emit_arbitration` verdict: `override` (machine value wrong
   → supplies `final_value`) or `confirm`, plus `confidence` and a `rationale`
   that must cite a diff line / traced symbol / evidence quote.

**Fable-specific API handling** (see `claude-api` skill): thinking is always on,
so the caller passes **no `thinking` parameter** and **no sampling params**
(`temperature`/`top_p`/`top_k` would 400). Structured output is forced via
`tool_choice` on the final turn.

**Three queues** (`--queues disagreements,pairs,audit`):
| queue | items | field arbitrated |
|---|---|---|
| `disagreements` | 183 Tier-2 double-annotation `kind` splits | `kind` |
| `pairs` | all 675 A/B regression pairs (逐条过目) | `evidence_tier` (can demote to C) |
| `audit` | deterministic 10% stratified-by-repo sample of perf-related cards | `kind` |

**Provenance (non-negotiable).** Every verdict is appended to
`cases/audits/audit_log.jsonl` with **`auditor: "fable-5"`** (never `"rui"`), plus
a `provenance` block (model id, `prompt_version`, run id, token counts,
`source_queue`). The machine annotations in `cards.jsonl` / `regression_pairs.jsonl`
are **never edited** — the arbiter's decisions live only in the append-only log.

**Layering / consumption.** `scripts/apply_audits.py` folds the log into
`cases/cards_final.jsonl`: an `override` on `(target_id, field)` wins over the
machine value; latest override by timestamp wins. Because the log is append-only
and timestamp-ordered, a later **human** pass (if ever done) supersedes Fable's
verdicts without redoing anything.

**Idempotent / cost-capped / concurrent.** Re-running skips `(target_id, field)`
pairs already in the log; `--max-cost-usd` stops cleanly; N workers.

## Cost

Measured token profile → ~$150 for all three queues at $10/$50 (upper bound ~$600
if full-diff context / multi-vote). Well within the $6,000 cap. Per-run spend is
recorded in `reports/arbitration_run.md`.

## ⚠️ Current blocker (as of 2026-07-16)

Fable 5 is **not runnable on this Bedrock account**: every request returns
`400 - data retention mode 'default' is not available for this model`. Fable
requires 30-day data retention; this account is on `default` retention, an
org/account-level setting that cannot be changed from the instance. The pipeline
(`arbitrate_fable.py`) is built, validated in shape, and ready; it will run
unchanged the moment either (a) the account's Bedrock data-retention mode is set
to 30-day for Fable, or (b) a Fable-capable credential/region is provided. Until
then, arbitration has not executed and `audit_log.jsonl` contains no fable-5
records. See `reports/BLOCKERS.md`.
