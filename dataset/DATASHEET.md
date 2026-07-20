# Datasheet — MegaPerfBench (Phase 0)

Following *Datasheets for Datasets* (Gebru et al.). Records provenance, license,
construction, and known biases so downstream use is informed and the eventual
publish/no-publish decision is well-founded. Phase 0 in progress; sections marked
*(pending)* fill as stages complete.

## Motivation

- **Purpose.** Build the foundation dataset for a perf-regression-detection agent
  for ML-infra codebases: labeled perf-bug case cards + inducing/fix regression
  pairs + a frozen evaluation protocol. No public "ML-infra performance-regression
  inducing-commits" dataset exists; this is itself a contribution.
- **Gap it fills.** GitHub Advisory DB covers only security (CIA), not perf. Prior
  perf-bug taxonomies (TF/PyTorch 12+19; DeepSpeed/Megatron/Colossal 34 symptoms)
  studied issues, not commit-level inducing/fix pairs with manifest conditions.

## Composition

- **Instances.** (1) case cards = commits judged for perf relevance/kind/symptom/
  mechanism/detectability; (2) regression pairs = (inducing, fix) with magnitude
  and manifest conditions; (3) negatives (three tiers).
- **Source population.** 33,089 commits (full history to 2026-07-13/14):
  Megatron-LM 9,214 · vLLM 18,703 · DeepSpeed 3,225 · TransformerEngine 1,947.
  Plus vLLM `performance`-labeled issues (503), reverts, Megatron golden-values
  churn (323 commits; 37/332 iteration-time-gated test cases), and academic seeds.
- **Labels.** Machine (Opus 4.8) with mandatory evidence citations, then
  **machine arbitration by Fable 5** layered append-only on top (`cases/audits/`);
  machine labels never overwritten. **v0.1 has NO human ground truth** — see the
  arbitration bias entry below and `reports/ARBITRATION.md`.
- **Splits.** Temporal (train < 2026-01-01; dev/test after) + a repo-holdout view.
  See `splits/protocol.md`.

## Collection process

- **Method.** Three-tier funnel (deterministic routing → non-agentic Opus-4.8
  screening → agentic deep read) + agentic SZZ (3-vote) for inducing commits +
  four side channels (issues/reverts/golden-values/academic). Full method in
  `intro.md` and the per-stage `reports/`.
- **Tools.** Local full git clones (offline `git show`/`git log -S`); Bedrock
  Opus 4.8; forced-tool structured output for schema validity; on-disk GitHub API
  cache (`raw/gh_cache/`) for PR/issue text.
- **Calibration.** Tier-1 recall gated ≥95% on a 48-card pilot gold set (29
  positives, 6 regression-fix); measured recall = **1.000** (6/6 regression-fix)
  leakage-free. Report: `reports/tier1_calibration.md`.
- **Timeframe.** Commit history spans 2019-03 (Megatron) → 2026-07; dataset built
  2026-07.

## Upstream licenses (all permissive; recorded for a possible future release)

| repo | license |
|---|---|
| Megatron-LM (NVIDIA) | BSD-3-Clause (with embedded Apache-2.0 portions) |
| vLLM | Apache-2.0 |
| DeepSpeed (Microsoft) | Apache-2.0 |
| TransformerEngine (NVIDIA) | Apache-2.0 |

Derived data stores commit SHAs, quoted diff/PR snippets as evidence, and machine
labels. Data is kept on EFS only and not redistributed; any release will re-verify
license compatibility for quoted upstream text.

## Known biases & limitations

- **Screen-bucket ≠ full history.** Tier-0 drops docs/format/empty-merge/CI-only
  commits from the LLM path (25,759 of 33,089 screened). Perf bugs hidden entirely
  in those buckets are out of scope by construction (CI-only → S4 instead).
- **Recall-first thresholds.** Tier-1 is tuned to maximize recall (a miss is
  unrecoverable); it over-includes, and the false-positive load is absorbed by
  Tier 2. Positive rates here are NOT population base rates.
- **SZZ noise.** Perf fixes often don't touch the inducing line (fix-forward,
  cache added elsewhere); mitigated by 3-vote SZZ + explicit regression-vs-latent
  split + human review of all pairs, but B-tier pairs carry residual attribution
  risk.
- **Golden-values confounding.** Container bumps / GB200 onboarding cause large
  iteration-time swings unrelated to code; these are marked `env_confounded` and
  excluded from pairs (see `reports/s4_golden_values.md`).
- **Model memorization.** These are public commits; a memorization check (§8.7.3)
  flags pairs the bare model can recall, and headline metrics are reported on the
  novel subset.
- **Single-annotator model.** All machine labels from one model family (Opus 4.8);
  20% double-annotation + Fable-5 arbitration bound the agreement, but systematic
  model blind spots are a residual risk.
- **★ Machine arbitration, not human (v0.1's central caveat).** Rui delegated all
  §6 arbitration to **Fable 5** (`claude-fable-5`) rather than doing it by hand
  (decision 2026-07-16). Therefore **the dataset is fully machine-labeled and
  machine-arbitrated — there is no human ground truth in v0.1**. The Tier-2 κ=0.81
  is Opus-vs-Opus; the arbitration on top is Fable-vs-Opus (a stronger, different
  model, run blind-first) — both machine–machine. Do NOT describe any subset as
  "human-verified"/"gold" in the human sense. Verdicts carry `auditor: "fable-5"`
  in `cases/audits/audit_log.jsonl`; the log is append-only so a later human pass
  can supersede them without rework. Full mechanics + provenance in
  `reports/ARBITRATION.md`.

## Deviations from the plan (`intro.md`), with rationale

1. **Repos re-cloned.** Plan assumed existing clones; none were present. Fresh
   full clones; counts match (33,089 vs 33,085; +4 new commits).
2. **`screen` bucket 25,759 (plan est. 28–30k).** Tier-0 additionally carves out
   `skip-empty` (merges) and `smoke-ci`; lowers cost, approved by Rui.
3. **Tier-1 channel = standard API, not batch.** Bedrock Batch does not support
   Opus 4.x on this account (Sonnet/Haiku 4.5 only). To keep the "all LLM calls
   use Opus 4.8" rule, Tier 1 runs on the standard API (~$1,043, under the $6,000
   cap). Decision + evidence: `reports/tier1_channel_decision.md`.
4. **Pilot data path.** Plan pointed at a macOS path; Rui transferred the pilot
   bundle to `raw/pilot/` (option A). Prompt bumped to `tier1_screen.v2` with real
   gold few-shots.
5. **Local-first side channels; S2 issues skipped.** Rui set "stay local, no
   GitHub". Tier 2/3 run on local git only (`fetch_pr` degrades gracefully);
   the S2 vLLM-issue channel, which genuinely needs the GitHub API, is skipped.
6. **§6 arbitration by Fable 5, not human.** See the "Machine arbitration" bias
   entry above + `reports/ARBITRATION.md`. This is the load-bearing deviation for
   any downstream "gold"/κ claim.

## Maintenance

- Versioned by local git in the dataset root (no remote — by rule). Schemas and
  prompts are version-stamped; every annotation records the `prompt_version` used.
- Cost monitored per stage; $4,500 warn line, $6,000 hard cap.

## v0.2 additions — method & limitations (dataset-complete)

v0.1 (`git tag v0.1`) is the frozen evaluation core; v0.2 enhances it in new
files (frozen files byte-immutable, verified). Final numbers: `reports/dataset_stats.md`.

- **S2 issues.** 880 GitHub issues (vLLM `performance`-label + Megatron/DeepSpeed
  keyword hits) fetched once (host-side, authenticated) and deep-read offline with
  Opus 4.8 (`s2_deepread.v1`, schema `s2_issue.v1`). User-measured magnitude and
  reproduction configs are backfilled onto cards only on **exact** PR/sha match
  (`s2_join`); fuzzy matches parked in `join_review.jsonl`, not auto-joined.
  Limitation: S2 is an enhancement stream, single-annotated (no double-label).
- **Memorization probe.** Bare Opus 4.8 (no tools/diff/context) on every test-split
  pair + 10% of test perf cards; **0/196 flagged memorized** → the test set is not
  contaminated and headline metrics may be reported on the full test set. Side
  annotation (`splits/memorization_flags.jsonl`); splits unchanged.
- **Taxonomy (`taxonomy.v1`).** Bottom-up open-coding over 600 stratified perf
  cards → 256 raw leaves → iterative LLM consolidation → **11 categories / 74
  mechanism-based leaves**. All 5,016 perf cards labeled (forced choice); QA 10%
  double-label **κ=0.971**, other-unclear **0.02%**. Limitation: 74 leaves exceeds
  the 25–55 target — the model resisted merging genuine mechanism distinctions
  further; accepted per the work-book's "don't hard-freeze" rule. Taxonomy is
  machine-induced and machine-applied (see the machine-arbitration caveat above —
  the same "no human ground truth" applies to the taxonomy).
- **Dedup links.** Same-root cards/pairs are *linked* (`related_cases` /
  `related_pairs`), never deleted — 150 cards, 90 pairs.
- **Validation.** `scripts/validate_dataset.py`: schema, referential integrity,
  split disjointness, frozen-file byte-identity — 10/10 pass
  (`reports/final_validation.md`). Two frozen-v0.1 pairs have unresolvable inducing
  shas (SZZ/S2 shorthand): recorded as a known limitation, not edited.
- **Still future work:** a human-anchored gold subset and a GPU-reproducible
  execution subset — neither exists in v0.2.
