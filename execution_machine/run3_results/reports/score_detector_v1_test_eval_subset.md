# Score — detector_v1 on test_eval_subset (metrics.v1)

- generated: 2026-07-19T15:40:57.155400+00:00 · **static ceiling 57.1%** · judge us.anthropic.claude-opus-4-8
- positives 934 · negatives 954 · leak_attempts 0 ✓
- judge spend ~$18.23

## recall@budget (severity∈{critical,important}, ceiling 57.1%)

| budget | recall |
|--:|---|
| 1 | 173/934 (18.5%) |
| 2 | 178/934 (19.1%) |
| 5 | 178/934 (19.1%) |

## per-kind recall@2 (**regression-fix = north star, n≈145**)

| kind | recall@2 |
|---|---|
| optimization | 126/615 (20.5%) |
| regression-fix **★** | 33/261 (12.6%) |
| config-default-change | 13/38 (34.2%) |
| perf-infra-or-test | 2/9 (22.2%) |
| not-perf | 4/9 (44.4%) |
| unclear | 0/2 (0.0%) |

## per-taxonomy recall@2

| category | recall@2 |
|---|---|
| kernel-efficiency | 97/319 (30.4%) |
| compilation | 15/104 (14.4%) |
| memory-management | 8/92 (8.7%) |
| host-overhead | 11/85 (12.9%) |
| config-observability | 16/78 (20.5%) |
| concurrency-sync | 7/61 (11.5%) |
| collective-comm | 5/59 (8.5%) |
| inference-serving | 3/44 (6.8%) |
| memory-footprint | 11/41 (26.8%) |
| io-startup | 2/26 (7.7%) |
| parallelism-scheduling | 3/24 (12.5%) |
| n/a | 0/1 (0.0%) |

## benign FPR (severe & conf≥0.5), Wilson 95% CI

| negative_type | FP/N | rate | 95% CI |
|---|---|--:|---|
| false-signal-perf-infra | 5/42 | 11.9% | [5.2, 25.0] |
| false-signal-smoke-ci | 8/150 | 5.3% | [2.7, 10.2] |
| hard-negative-hotfile | 105/459 | 22.9% | [19.3, 26.9] |
| hard-negative-lookalike | 1/3 | 33.3% | [6.1, 79.2] |
| random-benign | 23/300 | 7.7% | [5.2, 11.2] |
| _weighted_total | 142/954 | 14.9% | [12.8, 17.3] |
