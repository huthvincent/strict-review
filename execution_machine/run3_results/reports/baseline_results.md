# Baseline scorecard — test_eval_subset (Stage 2, metrics.v1)

- generated: 2026-07-19T06:57:39.379984+00:00 · **static ceiling 57.1%** · judge us.anthropic.claude-opus-4-8 / eval_judge.v1 (κ=0.926)
- contestants: (a) Megatron-strict, (b) generic, (c) keyword, (d) cross-family (Nova Pro)
- 全部跑在同一冻结子集（**1888 唯一 item** = 818 perf-pos + 116 pairs + 954 uniq neg）；`leak_attempt` 累计 = 0 ✓
- **子集去重说明**：源 test.txt 含 41 个重复 hard-negative-hotfile 行（v0.2 数据既有伪影），
  已去重；hotfile 负样本因此为 459（非目标 500），已记录，不影响其余配额与结论。
- **per-kind 分母说明**：regression-fix 的 261 = case 侧 regression-fix 卡 + 116 回归对（pairs）合并；
  北极星口径 n≈145 指 case 侧 regression-fix。恒等 39/261 在 (a)(b) 属巧合（judge 独立打分）。

## recall@budget（severity∈{critical,important}）

| baseline | recall@1 | recall@2 | recall@5 | leak |
|---|---|---|---|--:|
| (a) Megatron-strict | 89/934 (9.5%) | 122/934 (13.1%) | 127/934 (13.6%) | 0 |
| (b) generic | 101/934 (10.8%) | 114/934 (12.2%) | 118/934 (12.6%) | 0 |
| (c) keyword | 0/934 (0.0%) | 0/934 (0.0%) | 0/934 (0.0%) | 0 |
| (d) cross-family | 57/934 (6.1%) | 61/934 (6.5%) | 61/934 (6.5%) | 0 |

## per-kind recall@2（**regression-fix = 北极星, n≈145**）

| baseline | **regression-fix** | config-default-change | not-perf | optimization | perf-infra-or-test | unclear |
|---|---|---|---|---|---|---|
| (a) Megatron-strict | **39/261 (14.9%)** | 16/38 (42.1%) | 2/9 (22.2%) | 61/615 (9.9%) | 4/9 (44.4%) | 0/2 (0.0%) |
| (b) generic | **39/261 (14.9%)** | 10/38 (26.3%) | 2/9 (22.2%) | 61/615 (9.9%) | 2/9 (22.2%) | 0/2 (0.0%) |
| (c) keyword | **0/261 (0.0%)** | 0/38 (0.0%) | 0/9 (0.0%) | 0/615 (0.0%) | 0/9 (0.0%) | 0/2 (0.0%) |
| (d) cross-family | **21/261 (8.0%)** | 14/38 (36.8%) | 2/9 (22.2%) | 21/615 (3.4%) | 3/9 (33.3%) | 0/2 (0.0%) |

## per-taxonomy recall@2（11 大类）

| baseline | collective-comm | compilation | concurrency-sync | config-observability | host-overhead | inference-serving | io-startup | kernel-efficiency | memory-footprint | memory-management | n/a | parallelism-scheduling |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) Megatron-strict | 10/59 (16.9%) | 11/104 (10.6%) | 11/61 (18.0%) | 15/78 (19.2%) | 11/85 (12.9%) | 4/44 (9.1%) | 6/26 (23.1%) | 35/319 (11.0%) | 7/41 (17.1%) | 9/92 (9.8%) | 0/1 (0.0%) | 3/24 (12.5%) |
| (b) generic | 10/59 (16.9%) | 15/104 (14.4%) | 11/61 (18.0%) | 10/78 (12.8%) | 9/85 (10.6%) | 3/44 (6.8%) | 5/26 (19.2%) | 35/319 (11.0%) | 7/41 (17.1%) | 7/92 (7.6%) | 0/1 (0.0%) | 2/24 (8.3%) |
| (c) keyword | 0/59 (0.0%) | 0/104 (0.0%) | 0/61 (0.0%) | 0/78 (0.0%) | 0/85 (0.0%) | 0/44 (0.0%) | 0/26 (0.0%) | 0/319 (0.0%) | 0/41 (0.0%) | 0/92 (0.0%) | 0/1 (0.0%) | 0/24 (0.0%) |
| (d) cross-family | 5/59 (8.5%) | 5/104 (4.8%) | 6/61 (9.8%) | 8/78 (10.3%) | 2/85 (2.4%) | 2/44 (4.5%) | 5/26 (19.2%) | 13/319 (4.1%) | 4/41 (9.8%) | 9/92 (9.8%) | 0/1 (0.0%) | 2/24 (8.3%) |

## benign FPR（severe & conf≥0.5），Wilson 95% CI

| baseline | false-signal-perf-infra | false-signal-smoke-ci | hard-negative-hotfile | hard-negative-lookalike | random-benign | _weighted_total |
|---|---|---|---|---|---|---|
| (a) Megatron-strict | 5/42 (12%, [5,25]) | 13/150 (9%, [5,14]) | 108/459 (24%, [20,28]) | 1/3 (33%, [6,79]) | 37/300 (12%, [9,17]) | 164/954 (17%, [15,20]) |
| (b) generic | 1/42 (2%, [0,12]) | 1/150 (1%, [0,4]) | 32/459 (7%, [5,10]) | 1/3 (33%, [6,79]) | 11/300 (4%, [2,6]) | 46/954 (5%, [4,6]) |
| (c) keyword | 7/42 (17%, [8,31]) | 10/150 (7%, [4,12]) | 20/459 (4%, [3,7]) | 1/3 (33%, [6,79]) | 6/300 (2%, [1,4]) | 44/954 (5%, [3,6]) |
| (d) cross-family | 12/42 (29%, [17,44]) | 33/150 (22%, [16,29]) | 125/459 (27%, [23,31]) | 2/3 (67%, [21,94]) | 52/300 (17%, [13,22]) | 224/954 (23%, [21,26]) |

## 成本 / 延迟 / 判定开销

| baseline | $/item | s/item | total detect $ | judge $ |
|---|--:|--:|--:|--:|
| (a) Megatron-strict | $0.347 | 25.5 | $655.61 | $17.80 |
| (b) generic | $0.245 | 18.1 | $462.07 | $18.01 |
| (c) keyword | $0.000 | 0.0 | $0.00 | $16.01 |
| (d) cross-family | $0.000 | 1.7 | $0.00 | $18.34 |

## Stage-2 GATE — baseline (a) Megatron-strict 的 regression-fix recall@2

- regression-fix recall@2 = 39/261 = **14.9% ∈ [5,45%] → 正常，继续 Stage 3**

- 合计：detect ~$1117.68 + judge ~$70.16 = **~$1187.84**
- leak_attempts 合计：0（须为 0）

