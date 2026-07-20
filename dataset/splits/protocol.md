# MegaPerfBench — Evaluation Protocol (Core Output 3)

> Frozen contract for how any perf-regression detector is evaluated on this
> dataset. Written in Phase 0 per intro.md §8.7 ("协议先写死"); the baseline
> *numbers* are filled in Phase 2, but every rule below is fixed now and must
> not change after `splits/FROZEN` exists. Version: `protocol.v1`.

This document specifies the five things §8.7 requires: (1) split rules,
(2) leakage-prevention presentation rules, (3) memorization check, (4) metric
definitions, (5) baseline definitions.

---

## 1. Split rules

Units. The evaluation unit is a **case** (`case_id`, a commit) for
detection/classification metrics, and a **regression pair** (`pair_id`) for the
inducing-commit-attribution metric. A pair's `fix_sha` case and its `inducing_sha`
context are always assigned to the SAME split as the pair (no pair straddles a
split boundary).

Primary split = **temporal** (prevents trivially learning from the future):

| split | rule | purpose |
|---|---|---|
| `train` | commit author-date **< 2026-01-01** | knowledge base / few-shot / rule mining |
| `dev`   | date **≥ 2026-01-01**, first half of the post-cutoff window by date | threshold & prompt tuning |
| `test`  | date **≥ 2026-01-01**, second half | frozen final evaluation |

The dev/test halving is by the **median commit date** of the post-2026-01-01
positive set, computed once at freeze time and recorded in `splits/split_stats.json`
(so it is reproducible, not re-derived each run).

Secondary split = **repository hold-out** (generalization view, reported
alongside but not the headline): additionally produce a report where `train`
**excludes Megatron-LM entirely** and we measure test performance **on Megatron-LM**
cases. This answers "does a detector trained on vLLM/DeepSpeed/TE transfer to an
unseen repo?" Output: `reports/split_repo_holdout.md` (Phase 2).

Negatives. §5 negatives (random-benign / hard-negatives / false-signal) are
assigned to splits by the **same temporal rule** as cases, and are mixed into
dev/test at the natural base rate observed in Tier-0/Tier-1 (so the false-positive
metric reflects reality, not a balanced toy set). The exact negative:positive
ratio per split is recorded in `split_stats.json`.

Files. `splits/train.txt`, `dev.txt`, `test.txt` — one `case_id` or `pair_id`
per line. `splits/FROZEN` (empty sentinel) appears when the split is locked;
after that, edits are forbidden (enforced by convention + the local-git tag `v0.1`).

---

## 2. Leakage-prevention presentation rules (the core integrity rule)

When the detector is evaluated on a case/pair, its input is **exactly the
information available at the inducing commit's point in time** and nothing after:

- ALLOWED: the inducing commit's `(diff, message, and the repository tree snapshot
  at its parent)`. I.e. what a reviewer would see at PR time.
- FORBIDDEN as input: the fix commit, any issue/PR discussion, golden-value
  changes, revert, or any commit dated after the inducing commit. These are the
  *labels/evidence*, never detector inputs.

Concretely, the evaluation harness constructs each test item from the
`inducing_at_time_context` (§8.5) — `parent_sha` + inducing `diff` + `message` —
and MUST NOT pass any card field derived from later signals (`magnitude_reported`,
`issue_refs`, `pr_ref`, `inducing_ref`, fix `mechanism`). A detector that reads
those is disqualified. The harness enforces this by only ever handing the detector
a rebuilt "PR-time view", never the card itself.

For pure detection on a single commit (not a pair), the same rule holds: the
detector sees only that commit's diff/message + parent snapshot.

---

## 3. Memorization check (§8.7.3)

Model weights may already "know" these public commits. For every `test` pair,
before running any detector, ask **bare Opus 4.8** (no repo access, no tools):
"Given this commit `{repo}@{sha}` with subject `{subject}` — was it later found
to cause a performance problem, and if so what/how?" (prompt frozen in
`prompts/memorization_probe.v1.md`). If the bare model can produce the specific
symptom/mechanism/inducing-link from memory alone, the pair is flagged
`memorized: true`.

Reporting: results split into `memorized` vs `novel` subsets; the **headline
metric is reported on the `novel` subset**, with the full-set number reported
alongside and the memorized fraction stated explicitly. A detector that only
wins on memorized items is not credited with generalization.

---

## 4. Metric definitions

Let a "PR-comment budget" model the real reviewer constraint (the baseline to
beat, Megatron's claude[bot], comments on a bounded number of things per PR).

- **recall@budget=2**: per PR (or per commit), the detector may surface at most 2
  findings; a positive case counts as caught if a correct finding is among its 2.
  This is the north-star proxy (§8.7.4). Report at budget ∈ {1, 2, 5} for a curve.
- **per-kind recall**: recall computed separately for `regression-fix`,
  `optimization`, `config-default-change` (the three that matter), so a detector
  can't hide a weakness on regressions behind easy optimizations.
- **per-taxonomy recall**: same, bucketed by Phase-1 `taxonomy_label` (filled once
  the taxonomy exists; left as a stub key now).
- **benign false-positive rate**: fraction of §5 negatives on which the detector
  raises a finding, measured separately on each negative tier (random-benign /
  hard-negative / false-signal). The hard-negative FPR is the honest one.
- **inducing-commit attribution accuracy** (pairs only): for detectors that also
  localize, fraction of pairs whose predicted inducing commit == the labeled one
  (exact sha), reported separately for evidence-tier A vs B pairs.
- **static ceiling (honesty bound)**: the fraction of test positives with
  `static_detectability == "low"`. No static/diff-only detector can exceed
  `1 − low_fraction` recall; this is reported as the ceiling above every recall
  number so results are not oversold.

All metrics are computed by `scripts/eval_metrics.py` (Phase 2) from the frozen
splits + a detector's predictions file; the metric code is versioned with the
protocol.

---

## 5. Baseline definitions (numbers filled in Phase 2)

Two baselines are run **unchanged** on the frozen `test` set, before any detector
development, to establish the bar (§8.7.5, §11 Phase 2):

- **(a) Megatron strict-review**: NVIDIA/Megatron-LM's current
  `.github/workflows/claude_review.yml` `/claude strict-review` rubric prompt,
  run verbatim (dtype / TP-PP-SP-CP-EP / `.item()` sync / redundant collectives /
  CUDA-graph compatibility). This is the incumbent we must beat.
- **(b) bare Opus 4.8 generic review**: a plain "review this diff for performance
  problems" prompt, no dataset knowledge.

Both are evaluated under the identical presentation rules (§2) and metrics (§4);
results table → `reports/baseline_results.md`. **Without this step no later
"better" is defensible (§11).**

---

## Freeze checklist (Phase 0 §8.8 "冻结")

- [ ] `train.txt` / `dev.txt` / `test.txt` written from `cases/cards_final.jsonl`
      + `pairs/regression_pairs.jsonl` by `scripts/make_splits.py`
- [ ] `splits/split_stats.json` (median date, per-split counts, neg:pos ratios)
- [ ] `prompts/memorization_probe.v1.md` frozen
- [ ] `splits/FROZEN` sentinel created
- [ ] local git tag `v0.1`

*This protocol is `protocol.v1`. Any change before freeze must bump the version
and be noted in the changelog; no change after `FROZEN`.*
