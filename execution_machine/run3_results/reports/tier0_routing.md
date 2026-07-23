# Tier 0 — Deterministic Routing Report

- generated: 2026-07-14T02:00:22.817277+00:00
- router_version: `tier0.v1`
- total commits routed: **33089** (all sha have a routing record — DoD met)

## Bucket distribution (all repos)

| bucket | count | share | meaning |
|---|---:|---:|---|
| `screen` | 25759 | 77.8% | → Tier 1 LLM screening |
| `smoke-ci` | 2070 | 6.3% | → S4 golden/CI sub-pipeline |
| `skip-docs` | 2562 | 7.7% | docs/assets only (dropped from LLM path) |
| `skip-format` | 42 | 0.1% | whitespace-only diff (git -w empty) |
| `skip-empty` | 2656 | 8.0% | merge/empty (no first-parent diff) |

## Per-repo breakdown

| repo | `screen` | `smoke-ci` | `skip-docs` | `skip-format` | `skip-empty` | total |
|---|---:|---:|---:|---:|---:|---:|
| Megatron-LM | 5451 | 808 | 301 | 14 | 2640 | 9214 |
| vllm | 16251 | 979 | 1436 | 21 | 16 | 18703 |
| DeepSpeed | 2301 | 225 | 695 | 4 | 0 | 3225 |
| TransformerEngine | 1756 | 58 | 130 | 3 | 0 | 1947 |

## Signal flags (count of commits touching each; for §5 negatives & S4)

| signal | commits |
|---|---:|
| `touches_golden` | 415 |
| `touches_ci` | 2945 |
| `touches_benchmark` | 901 |
| `test_only` | 1953 |
| `touches_core_code` | 22757 |

## Notes / deviations from plan

- Plan §3 estimated `screen` ≈ 28–30k; actual **25759**. The gap is legitimate: the router additionally carves out `skip-empty` (merge commits with no first-parent diff, mostly Megatron) and `smoke-ci`, which the plan folded into the screen estimate. Net effect **lowers** Tier-1 batch cost.
- `skip-format` is intentionally tiny/conservative: whitespace-only via `git log -w`, NOT import reordering — a false skip is unrecoverable lost recall (§10), a false `screen` costs a few cents in Tier 1.
- `smoke-ci` = **2070** commits feed the S4 sub-pipeline (golden-values / CI / benchmark churn).
