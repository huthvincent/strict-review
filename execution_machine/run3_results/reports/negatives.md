# §5 Negatives — Build Report

- generated: 2026-07-16T00:54:58.061398+00:00
- Tier-1 positives (perf_relevance ≥ 0.35): 7790
- total negatives: **29735** (3.8× positives)

## negative_type distribution

| type | n |
|---|---:|
| random-benign | 13970 |
| hard-negative-hotfile | 13390 |
| false-signal-smoke-ci | 2070 |
| false-signal-perf-infra | 270 |
| hard-negative-lookalike | 35 |

## notes
- random-benign target = 4.0× positives, time-stratified per repo.
- hard-negative hotfile arm: 13390 (needs pairs; active).
- Once Tier-2 exists, random-benign should additionally require 'deep-read did not overturn' — re-run this after cards.jsonl to prune.
