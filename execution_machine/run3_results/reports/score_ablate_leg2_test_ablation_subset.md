# Score — ablate_leg2 on test_ablation_subset (metrics.v1)

- generated: 2026-07-19T15:53:18.300882+00:00 · **static ceiling 57.1%** · judge us.anthropic.claude-opus-4-8
- positives 934 · negatives 400 · leak_attempts 0 ✓
- judge spend ~$18.38

## recall@budget (severity∈{critical,important}, ceiling 57.1%)

| budget | recall |
|--:|---|
| 1 | 188/934 (20.1%) |
| 2 | 193/934 (20.7%) |
| 5 | 193/934 (20.7%) |

## per-kind recall@2 (**regression-fix = north star, n≈145**)

| kind | recall@2 |
|---|---|
| optimization | 133/615 (21.6%) |
| regression-fix **★** | 35/261 (13.4%) |
| config-default-change | 17/38 (44.7%) |
| perf-infra-or-test | 4/9 (44.4%) |
| not-perf | 4/9 (44.4%) |
| unclear | 0/2 (0.0%) |

## per-taxonomy recall@2

| category | recall@2 |
|---|---|
| kernel-efficiency | 98/319 (30.7%) |
| compilation | 19/104 (18.3%) |
| memory-management | 3/92 (3.3%) |
| host-overhead | 7/85 (8.2%) |
| config-observability | 20/78 (25.6%) |
| concurrency-sync | 11/61 (18.0%) |
| collective-comm | 8/59 (13.6%) |
| inference-serving | 5/44 (11.4%) |
| memory-footprint | 16/41 (39.0%) |
| io-startup | 1/26 (3.8%) |
| parallelism-scheduling | 5/24 (20.8%) |
| n/a | 0/1 (0.0%) |

## benign FPR (severe & conf≥0.5), Wilson 95% CI

| negative_type | FP/N | rate | 95% CI |
|---|---|--:|---|
| false-signal-perf-infra | 2/16 | 12.5% | [3.5, 36.0] |
| false-signal-smoke-ci | 3/65 | 4.6% | [1.6, 12.7] |
| hard-negative-hotfile | 43/195 | 22.1% | [16.8, 28.4] |
| hard-negative-lookalike | 2/2 | 100.0% | [34.2, 100.0] |
| random-benign | 12/122 | 9.8% | [5.7, 16.4] |
| _weighted_total | 62/400 | 15.5% | [12.3, 19.4] |
