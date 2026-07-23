# S4 — Golden-Values Churn (Megatron-LM)

- generated: 2026-07-14T03:18:17.457082+00:00  |  version: `s4_golden.v1`
- iteration-time-gated test cases: **37** (analyst: 37) ✓
- gated golden files scanned: 63
- change records (consecutive golden edits): 101
- >5% median iteration-time increases: 11
- **accepted-regression candidates** (>5% slowdown in a non-env rebaseline): **1**
- env-confounded records (container/nightly/weekly): 34
- baseline_values (perf-test) records: 11

## commit-class distribution

| class | n |
|---|---:|
| feature-or-other | 47 |
| rebaseline | 20 |
| bulk-nightly-weekly | 19 |
| env-bump | 15 |

## top 15 iteration-time jumps (by Δ%)

| Δ% | class | env? | test_case / hw | commit |
|---:|---|:--:|---|---|
| +59.6 | bulk-nightly-weekly | Y | gpt3_15b_8t_release_sm_gb200 / gb200 | `6657173fd7` Update golden values of weekly tests (#3 |
| +27.3 | feature-or-other | n | gpt_grpo_tp8tp4_pp1_ep8ep2_dp8_throughputtest / h100 | `36411ddff1` Reapply 3955c49ed9af5e5b38dccdd30c1323c0 |
| +25.5 | feature-or-other | n | t5_release / h100 | `f098fe8482` feat: long convergence resiliency for re |
| +17.2 | rebaseline | n | gpt3_moe_mcore_te_tp4_ep2_etp2_pp2_scoped_cudagraph / h100 | `69ba8096f0` ci(hotfix): Update golden values after 2 |
| +15.1 | env-bump | Y | mimo_vlm_pretrain_convergence_tp1_pp1_cp1_dp8_seq_packing / h100 | `5fe3f06658` chore: Update Docker image version to 26 |
| +12.7 | env-bump | Y | t5_release_sm / h100 | `a50252b6ad` Update goldens for weekly tests after py |
| +10.7 | env-bump | Y | gpt3_moe_mcore_te_tp4_ep2_etp2_pp2_scoped_cudagraph / gb200 | `64bee49c96` Update base image to nvcr.io/nvidia/pyto |
| +7.5 | bulk-nightly-weekly | Y | mixtral_8x22b_tp2pp8ep8vpp1_release / h100 | `6657173fd7` Update golden values of weekly tests (#3 |
| +6.4 | env-bump | Y | deepseek_proxy_fsdp_ep2_fsdp2 / gb200 | `64bee49c96` Update base image to nvcr.io/nvidia/pyto |
| +6.2 | feature-or-other | n | gpt3_mcore_te_tp2_pp2_ep4_etp1_memory_speed / h100 | `e8024d716f` ADLR/megatron-lm!4068 - Compute shared e |
| +5.6 | feature-or-other | n | deepseek_proxy_fsdp_ep2_fsdp2 / gb200 | `6e091e1b68` chore: Update transformer-engine depende |
| +5.0 | bulk-nightly-weekly | Y | gpt3_weekly_mcore_tp2_pp2_current_scaling_native_fp8_tp_pp_sp_tp_overlap / h100 | `6657173fd7` Update golden values of weekly tests (#3 |
| +5.0 | feature-or-other | n | gpt_grpo_tp4_pp1_dp2_8b_cudagraphs_throughput / h100 | `4e85d74ae9` fix: enforce correct pass thresholds for |
| +4.0 | env-bump | Y | mimo_vlm_pretrain_convergence_tp1_pp1_cp2_dp8 / h100 | `5fe3f06658` chore: Update Docker image version to 26 |
| +3.9 | bulk-nightly-weekly | Y | t5_release_sm / h100 | `6657173fd7` Update golden values of weekly tests (#3 |

## Notes
- Accepted-regression candidates → feed Tier-A pairs (measured slowdown), after human audit (§6). Env-confounded rows are kept but excluded from pairs (§10).
- Pure re-baselines carry NO inducing reference; `inducing_window` gives the temporal candidate range (commits between consecutive same-file golden edits) for later agentic attribution.
- baseline_values rows need a numeric throughput/latency diff (per-batch-size); flagged for agent follow-up (e.g. 6204b925f ~-6-9% tok/s re-baselined as noise).
