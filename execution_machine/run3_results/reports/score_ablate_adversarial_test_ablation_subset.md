# Score — ablate_adversarial on test_ablation_subset (metrics.v1)

- generated: 2026-07-19T16:33:12.792514+00:00 · **static ceiling 57.1%** · judge us.anthropic.claude-opus-4-8
- positives 934 · negatives 400 · leak_attempts 0 ✓
- judge spend ~$18.52

## recall@budget (severity∈{critical,important}, ceiling 57.1%)

| budget | recall |
|--:|---|
| 1 | 212/934 (22.7%) |
| 2 | 223/934 (23.9%) |
| 5 | 223/934 (23.9%) |

## per-kind recall@2 (**regression-fix = north star, n≈145**)

| kind | recall@2 |
|---|---|
| optimization | 152/615 (24.7%) |
| regression-fix **★** | 45/261 (17.2%) |
| config-default-change | 18/38 (47.4%) |
| perf-infra-or-test | 5/9 (55.6%) |
| not-perf | 3/9 (33.3%) |
| unclear | 0/2 (0.0%) |

## per-taxonomy recall@2

| category | recall@2 |
|---|---|
| kernel-efficiency | 112/319 (35.1%) |
| compilation | 21/104 (20.2%) |
| memory-management | 11/92 (12.0%) |
| host-overhead | 9/85 (10.6%) |
| config-observability | 20/78 (25.6%) |
| concurrency-sync | 13/61 (21.3%) |
| collective-comm | 12/59 (20.3%) |
| inference-serving | 5/44 (11.4%) |
| memory-footprint | 12/41 (29.3%) |
| io-startup | 5/26 (19.2%) |
| parallelism-scheduling | 3/24 (12.5%) |
| n/a | 0/1 (0.0%) |

## benign FPR (severe & conf≥0.5), Wilson 95% CI

| negative_type | FP/N | rate | 95% CI |
|---|---|--:|---|
| false-signal-perf-infra | 2/16 | 12.5% | [3.5, 36.0] |
| false-signal-smoke-ci | 4/65 | 6.2% | [2.4, 14.8] |
| hard-negative-hotfile | 47/195 | 24.1% | [18.6, 30.6] |
| hard-negative-lookalike | 1/2 | 50.0% | [9.5, 90.5] |
| random-benign | 13/122 | 10.7% | [6.3, 17.4] |
| _weighted_total | 67/400 | 16.8% | [13.4, 20.7] |
