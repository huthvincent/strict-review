# Tier 1 — Live Micro-Smoke

- generated: 2026-07-14T02:04:24.071351+00:00
- run_id: `tier1_smoke_2026-07-14T02:03:21.270094+00:00`
- prompt_version: `tier1_screen.v1` | model: Opus 4.8 (live standard channel)
- commits classified: **16**  |  measured spend: **$0.7817**

## Token / cost per commit (measured, exact from API usage)

- avg input tokens: **8667** (min 3646, max 13610)
- avg output tokens: **221**
- avg cost/commit (standard $5/$25): **$0.0489**
- avg cost/commit (batch 0.5x): **$0.0244**

## Projected full Tier-1 batch cost

- screen-bucket commits: **25759**
- projected batch cost: **$629**  (plan §9 estimate: ~$700 — within range)

## kind_guess distribution (smoke sample; diverse, NOT base rate)

| kind | n |
|---|---:|
| optimization | 8 |
| not-perf | 5 |
| perf-infra-or-test | 2 |
| regression-fix | 1 |

- needs_deep_read: 9/16 (56%)

## Sample verdicts

| perf | kind | symptom | deep | commit | subject |
|---:|---|---|---|---|---|
| 0.90 | optimization | gpu-util-or-bubble | True | `Megatron-LM@ce8865c6c1e3` | Add forward all-gather overlap (#5513) |
| 0.72 | regression-fix | memory | True | `Megatron-LM@6213cff61b78` | ADLR/megatron-lm!2508 - Disable the FP8 transpose cache when using tor |
| 0.05 | not-perf | n/a | False | `Megatron-LM@48a887fecd01` | Remove use of exec_module (#5744) |
| 0.05 | perf-infra-or-test | n/a | False | `Megatron-LM@81b2cb9098be` | test: Dont use `dist.destroy_process_group` |
| 0.85 | optimization | throughput | True | `vllm@21472f32ea6c` | add pad-aware swiglu limit kernel (#48287) |
| 0.10 | perf-infra-or-test | n/a | False | `vllm@8bff831f0aa2` | [Benchmark] Cleanup deprecated nightly benchmark and adjust the docstr |
| 0.55 | optimization | latency | True | `vllm@c4f5cd60dae3` | [1/N] Add dense MHA path for sparse MLA short sequences (#47327) |
| 0.15 | not-perf | n/a | False | `vllm@cc99baf14dac` | [Misc] Make timeout passable in init_distributed_environment (#24522) |
| 0.70 | optimization | latency | True | `DeepSpeed@429e2ad212c1` | Add Hybrid Engine rollout in DeepSpeed to support On-Policy Distillati |
| 0.90 | optimization | throughput | True | `DeepSpeed@7f26bb6ae47c` | faster allreduce with omp parallel for reduce kernel (#4049) |
| 0.30 | optimization | throughput | True | `DeepSpeed@104193b45fe7` | Add AutoEP + AutoTP parallel folding (#8064) |
| 0.10 | not-perf | n/a | False | `DeepSpeed@aebdfb3b9257` | Fix Bug in transform.cu (#3534) |
| 0.20 | not-perf | n/a | False | `TransformerEngine@6377ca161c0e` | [PyTorch] Fix GIL/refcount abort in Comm+GEMM overlap and NCCL-EP init |
| 0.70 | optimization | throughput | True | `TransformerEngine@21ec6e04b4bc` | change softmax_lse correction of CP to FP32 (#1546) |
| 0.72 | optimization | throughput | True | `TransformerEngine@b782ea6ed498` | TE EP integration to MoEBlock (#3116) |
| 0.05 | not-perf | n/a | False | `TransformerEngine@a3ba4dffaccc` | Fix cpp warnings (#1639) |
