# Tier 1 — Full Screen Summary

- generated: 2026-07-15T20:16:05.402980+00:00
- channel: **standard API**, model Opus 4.8, prompt `tier1_screen.v2`
- commits screened: **25759** / 25,759 screen bucket (COMPLETE)
- measured spend: **$1078.51** (in=188,266,781 out=5,487,057 tokens)
- operating threshold T=0.4: **7737** commits → Tier 2 (30.0% of screened)  [model's own needs_deep_read: 8561]

## kind_guess distribution

| kind | n | % |
|---|---:|---:|
| not-perf | 15534 | 60.3% |
| optimization | 4423 | 17.2% |
| perf-infra-or-test | 2217 | 8.6% |
| unclear | 1795 | 7.0% |
| regression-fix | 1072 | 4.2% |
| config-default-change | 718 | 2.8% |

## symptom_guess distribution

| symptom | n |
|---|---:|
| n/a | 17487 |
| throughput | 4488 |
| memory | 1260 |
| latency | 1246 |
| compile-or-startup-time | 615 |
| gpu-util-or-bubble | 492 |
| hang | 169 |
| startup-time | 2 |

## Tier-2 volume at T=0.4 (per repo)

| repo | screened | → deep read |
|---|---:|---:|
| Megatron-LM | 5451 | 1422 |
| vllm | 16251 | 4848 |
| DeepSpeed | 2301 | 630 |
| TransformerEngine | 1756 | 837 |

- **Total Tier-2 deep reads at T=0.4: 7737** (plan §9 estimate 3–5k).
- Threshold rationale: bottom of the calibration full-recall plateau (recall=1.000 on 29 gold positives, 6/6 regression-fix at T≤0.40). Raise T toward 0.40 if Tier-2 volume must be trimmed — still full recall.
