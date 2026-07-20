# Stage 6 — 真实新 commit 前瞻试运行 (§6)

- generated: 2026-07-19T18:33:13.772000+00:00 · 冻结 detector_v1 · budget=2 · leak_attempts 0
- 窗口 2026-06-15..2026-07-19：61 个**数据集未见**的 Megatron-LM 新 commit（post-07-01 gave 53 (<80); widened to 06-15 per §6）
- **触发率**：26/61 commit 被报了问题（43%）
- 腿分布：{'leg3': 15, 'leg2': 6, 'leg1': 8} · 路由 taxonomy 分布：{'blocking-d2h-sync': 2, 'low-precision-state': 2, 'correctness-forces-slow-path': 2, 'boolean-guard-misfire': 2, 'activation-recompute': 1, 'redundant-buffer-copy': 1, 'prefix-state-caching': 1, 'kernel-config-tuning': 1}

> 无标签 → 定性产出。top-20（按 confidence 降序）供 Rui 人工核验。

## Top-20 finding（人工核验清单）

### 1. `acc7e644572b` (conf 0.75) — 2026-07-17
- commit: fix bug where Gemma4 is not working with recompute_granularity = "full" (#5324)
- finding [important]: 此改动触及 activation-recompute（memory-footprint），该类问题历史上主要在 [MoE (shared experts, TE grouped MLP), optionally MLA, context parallelism (CP>1) for CP path; recompute_granularity=full, recompute_method=uniform, recompute_num_layers=1 for memory path; also relevant to MoE/EP and TP, MoE / expert parallelis
- category/leg: risk-route:activation-recompute · suggested_benchmark: gpt-perf-dp8 — activation/param memory; OOM threshold
- 人工核验: `[ ] 真问题  [ ] 误报  [ ] 需查证`

### 2. `4a1f74350e76` (conf 0.75) — 2026-07-17
- commit: Avoid extra MFSDP v2 model-weight sync memcpy (#5834)
- finding [important]: 此改动触及 redundant-buffer-copy（memory-management），该类问题历史上主要在 [TP (low latency TP case mentioned) (issue-measured), MoE/expert-parallel models; sharded branch requires distributed optimizer (main_param_sharded), CUDA graphs enabled (share_cudagraph_io_buffers) with transformer decoder layers; PP/VPP chu
- category/leg: risk-route:redundant-buffer-copy · suggested_benchmark: gpt-perf-dp8 — allocator/fragmentation; peak-mem under DP8
- 人工核验: `[ ] 真问题  [ ] 误报  [ ] 需查证`

### 3. `53a2dd56fec6` (conf 0.75) — 2026-07-15
- commit: IMA fix by making the copy of book keeping buffer to GPU blocking (#5715)
- finding [important]: 此改动触及 blocking-d2h-sync（concurrency-sync），该类问题历史上主要在 [inference dynamic batching (dynamic inference context), distributed multi-rank (uses all-gather/barrier across world_size); benefit scales with rank count, MoE with EP/TP; multi-local-expert (num_local_experts>1) path most affected; alltoall and 
- category/leg: risk-route:blocking-d2h-sync · suggested_benchmark: determinism-perf — stream/sync ordering; determinism harness exposes stalls
- 人工核验: `[ ] 真问题  [ ] 误报  [ ] 需查证`

### 4. `e5344abbdf81` (conf 0.75) — 2026-07-14
- commit: Mamba prefix caching fixes (#5502)
- finding [important]: 此改动触及 prefix-state-caching（inference-serving），该类问题历史上主要在 [DCP/PCP asserted world_size==1 for hybrid/sliding-window paths, tensor-parallel (TP>1) with MLA; also PP-aware per comment, multi-engine / data-parallel or cross-restart (inconsistent lora_int_id across engines)] 配置下显现；The diff fixes Mamba pr
- category/leg: risk-route:prefix-state-caching · suggested_benchmark: gpt-perf — decode throughput/latency (proxy via gpt-perf)
- 人工核验: `[ ] 真问题  [ ] 误报  [ ] 需查证`

### 5. `8bf73659f692` (conf 0.72) — 2026-07-15
- commit: Set num_splits to 0 for FA4 inference (#5804)
- finding [important]: 此改动触及 kernel-config-tuning（kernel-efficiency），该类问题历史上主要在 [TP8, TP4, MoE (OpenAI triton_kernels matmul_ogs)] 配置下显现；The change sets `num_splits=0` (auto/heuristic split selection) instead of the fixed `num_splits=1` for FA4 varlen inference attention kernels, except when batc
- category/leg: risk-route:kernel-config-tuning · suggested_benchmark: module_performance — per-module kernel microbench
- 人工核验: `[ ] 真问题  [ ] 误报  [ ] 需查证`

### 6. `a28ca480bb00` (conf 0.72) — 2026-07-14
- commit: Short-circuit condition to avoid copying from GPU memory in `ChainedOptimizer` (#5623)
- finding [important]: 此改动触及 blocking-d2h-sync（concurrency-sync），该类问题历史上主要在 [inference dynamic batching (dynamic inference context), distributed multi-rank (uses all-gather/barrier across world_size); benefit scales with rank count, MoE with EP/TP; multi-local-expert (num_local_experts>1) path most affected; alltoall and 
- category/leg: risk-route:blocking-d2h-sync · suggested_benchmark: determinism-perf — stream/sync ordering; determinism harness exposes stalls
- 人工核验: `[ ] 真问题  [ ] 误报  [ ] 需查证`

### 7. `a79f49d37bd9` (conf 0.72) — 2026-07-14
- commit: Delegate reasoning token retention to the chat template in multi-turn conversations (#5276)
- finding [important]: In megatron/rl/inference/megatron.py::base_generate (the per-request hot path for RL rollouts), the extra_body now unconditionally sets "return_raw_text": True. This flips on the server-side return_raw_text path in chat_completions.py, which calls tokenizer.detokenize(result["prompt_tokens"]) for EV
- category/leg: redundant-hot-path-work · suggested_benchmark: Measure chat/completions latency and coordinator CPU for RL rollouts with long prompts (e.g. 8k-token prompts) comparing return_raw_text=True vs False; assert per-request detokenize cost scales with prompt length.
- 人工核验: `[ ] 真问题  [ ] 误报  [ ] 需查证`

### 8. `bcf4c8fb5179` (conf 0.72) — 2026-07-14
- commit: Inference: Add load aware routing to prefix caching.  (#5607)
- finding [important]: The default coordinator routing policy is changed from FIRST_PREFIX_BLOCK to LOAD_BALANCED in three places simultaneously: `InferenceConfig.prefix_caching_coordinator_policy` (megatron/core/inference/config.py:302), the argparse default for `--inference-dynamic-batching-prefix-caching-coordinator-po
- category/leg: config-toggle-perf-feature · suggested_benchmark: DP=8 coordinator with prefix caching enabled, many requests sharing a long common prompt prefix; measure cross-rank prefix cache hit rate and prefill FLOPs/latency under default policy (LOAD_BALANCED) vs FIRST_PREFIX_BLOCK vs longest_prefix.
- 人工核验: `[ ] 真问题  [ ] 误报  [ ] 需查证`

### 9. `f258d4fa82de` (conf 0.70) — 2026-07-17
- commit: Test zero-CTA copy-engine all-gather (#5858)
- finding [important]: static rule 'full-cuda-sync-in-overlap-path' matched antipattern: Calling the global torch.cuda.synchronize() (full device barrier) to order work against a side stream, which serializes all compute/comm and defeats overlap; stream.wait_stream should be used instead. (matched: 'torch.cuda.synchronize
- category/leg: static-rule:full-cuda-sync-in-overlap-path
- 人工核验: `[ ] 真问题  [ ] 误报  [ ] 需查证`

### 10. `61f31145dfb4` (conf 0.70) — 2026-07-16
- commit: Allow parameterless FSDP root modules (#5711)
- finding [important]: static rule 'tensor-cpu-alloc-then-to-device' matched antipattern: Allocating a tensor on CPU (often pinned) and immediately copying it to the device with .to(...), instead of constructing it directly on the target device. (matched: 'torch.ones(1, 4)\n    expected_output = model(x).to(device')
- category/leg: static-rule:tensor-cpu-alloc-then-to-device
- 人工核验: `[ ] 真问题  [ ] 误报  [ ] 需查证`

### 11. `f8e1ac64b058` (conf 0.70) — 2026-07-16
- commit: Refactor data parallel coordinator to enable modular handlers (#5550)
- finding [important]: static rule 'numpy-index-gpu-tensor-d2h-sync' matched antipattern: Indexing a GPU tensor with a NumPy array (or CPU-derived index built via np.*) forces an implicit blocking host-to-device copy of the index array. (matched: 'rank_idxs = np.fromiter(row.keys(), dtype=np.intp)\n            present = n
- category/leg: static-rule:numpy-index-gpu-tensor-d2h-sync
- 人工核验: `[ ] 真问题  [ ] 误报  [ ] 需查证`

### 12. `d981f66bebc9` (conf 0.70) — 2026-07-15
- commit: Overlap FSDP communication with compute (#5719)
- finding [important]: static rule 'cuda-stream-in-init' matched antipattern: Allocating a fresh torch.cuda.Stream() inside __init__ (or per-object construction), so N objects create N CUDA streams instead of sharing one. (matched: 'torch.cuda.Stream()')
- category/leg: static-rule:cuda-stream-in-init
- 人工核验: `[ ] 真问题  [ ] 误报  [ ] 需查证`

### 13. `d981f66bebc9` (conf 0.70) — 2026-07-15
- commit: Overlap FSDP communication with compute (#5719)
- finding [important]: static rule 'tensor-cpu-alloc-then-to-device' matched antipattern: Allocating a tensor on CPU (often pinned) and immediately copying it to the device with .to(...), instead of constructing it directly on the target device. (matched: 'torch.ones(dim))\n        self.inner = nn.Linear(dim, dim, bias=Fa
- category/leg: static-rule:tensor-cpu-alloc-then-to-device
- 人工核验: `[ ] 真问题  [ ] 误报  [ ] 需查证`

### 14. `a3279434e64b` (conf 0.70) — 2026-07-14
- commit: Exercise nested MFSDP CUDA graph capture (#5796)
- finding [important]: static rule 'cuda-stream-in-init' matched antipattern: Allocating a fresh torch.cuda.Stream() inside __init__ (or per-object construction), so N objects create N CUDA streams instead of sharing one. (matched: 'torch.cuda.Stream()')
- category/leg: static-rule:cuda-stream-in-init
- 人工核验: `[ ] 真问题  [ ] 误报  [ ] 需查证`

### 15. `a3279434e64b` (conf 0.70) — 2026-07-14
- commit: Exercise nested MFSDP CUDA graph capture (#5796)
- finding [important]: static rule 'tensor-cpu-alloc-then-to-device' matched antipattern: Allocating a tensor on CPU (often pinned) and immediately copying it to the device with .to(...), instead of constructing it directly on the target device. (matched: 'torch.zeros(dim))\n        self.layers = nn.ModuleList([nn.Linear(
- category/leg: static-rule:tensor-cpu-alloc-then-to-device
- 人工核验: `[ ] 真问题  [ ] 误报  [ ] 需查证`

### 16. `3a253ac5c2d3` (conf 0.70) — 2026-07-14
- commit: Inference: Do not route pad/dummy tokens to any expert (#4922)
- finding [important]: static rule 'tensor-from-pylist-on-device' matched antipattern: Constructing a tensor from a Python list literal with a non-CPU device forces a CPU materialization plus a host-to-device copy on every call; use torch.zeros/ones/full/empty directly on the device instead. (matched: 'torch.tensor([count
- category/leg: static-rule:tensor-from-pylist-on-device
- 人工核验: `[ ] 真问题  [ ] 误报  [ ] 需查证`

### 17. `33706f968e0e` (conf 0.66) — 2026-07-14
- commit: Implement Quantile Balancing in MoE (#5349)
- finding [important]: The new `quantile_balancing` routing path (router.py) issues a blocking `torch.distributed.all_gather_into_tensor(full_logits, ...)` over `tp_cp_group` on EVERY microbatch forward when `gather_size > 1` (TP/CP > 1). This all-gather transfers the whole logits tensor `[local_num_tokens * gather_size, 
- category/leg: missing-comm-overlap · suggested_benchmark: MoE forward-pass latency with moe_router_load_balancing_type=quantile_balancing under TP=2/4 + CP, compared against aux_loss routing at the same config; measure router step time and the all_gather stall.
- 人工核验: `[ ] 真问题  [ ] 误报  [ ] 需查证`

### 18. `eadbaa6189fb` (conf 0.62) — 2026-07-13
- commit: Inference: Add profile endpoints to chat completions.  (#5611)
- finding [important]: 此改动触及 profiling-accounting-fix（config-observability），该类问题历史上主要在 [pipeline model parallelism (1F1B non-interleaved schedule); triggers when get_num_microbatches() < pipeline_model_parallel_world_size, pipeline-parallel (1F1B schedule), impacts model/tensor-parallel runs (all_reduce over model-paralle
- category/leg: risk-route:profiling-accounting-fix · suggested_benchmark: gpt-perf — config-gated regressions; run with feature toggled
- 人工核验: `[ ] 真问题  [ ] 误报  [ ] 需查证`

### 19. `33706f968e0e` (conf 0.60) — 2026-07-14
- commit: Implement Quantile Balancing in MoE (#5349)
- finding [important]: `qb_dual_update` performs a column-wise top-k over the FULL gathered token dimension: `(scores - alpha).topk(col_target + 1, dim=0)` where `col_target = num_tokens * k // num_experts`. A `dim=0` top-k across all tokens (potentially tens of thousands of rows) selecting a large k is a costly sort-like
- category/leg: suboptimal-kernel-gate · suggested_benchmark: Microbenchmark qb_dual_update vs standard compute_topk on scores of shape [global_seq_len, num_experts] for representative num_tokens (e.g. 8192-65536) and num_experts (e.g. 64-256); isolate the dim=0 topk cost.
- 人工核验: `[ ] 真问题  [ ] 误报  [ ] 需查证`

### 20. `ecf55477df08` (conf 0.55) — 2026-07-15
- commit: chore(tests): AUT-851 move NCCL defaults from run_ci_test.sh to conftest (#5826)
- finding [important]: 此改动触及 mismatched-collective-config（collective-comm），该类问题历史上主要在 [multi-node data/model parallel (global world size > single-node GPU count), Megatron-FSDP (sharded data parallel with reduce-scatter grad reduction), distributed optimizer with DP+CP; sharded main_params unevenly distributed across rank
- category/leg: risk-route:mismatched-collective-config · suggested_benchmark: gpt-perf — multi-node TP/PP; watch all-reduce/all-gather time
- 人工核验: `[ ] 真问题  [ ] 误报  [ ] 需查证`

## 全部触发 commit（简表）

| sha | date | #findings | top severity | subject |
|---|---|--:|---|---|
| b4ad280d352e | 2026-07-17 | 1 | important | Route Lion through DistributedOptimizer and support single-m |
| acc7e644572b | 2026-07-17 | 1 | important | fix bug where Gemma4 is not working with recompute_granulari |
| f258d4fa82de | 2026-07-17 | 1 | important | Test zero-CTA copy-engine all-gather (#5858) |
| 4a1f74350e76 | 2026-07-17 | 1 | important | Avoid extra MFSDP v2 model-weight sync memcpy (#5834) |
| 61f31145dfb4 | 2026-07-16 | 1 | important | Allow parameterless FSDP root modules (#5711) |
| 740c16e6b80a | 2026-07-16 | 1 | suggestion | Inference: Extend default cuda-graph coverage to 512 tokens  |
| f8e1ac64b058 | 2026-07-16 | 1 | important | Refactor data parallel coordinator to enable modular handler |
| d981f66bebc9 | 2026-07-15 | 2 | important | Overlap FSDP communication with compute (#5719) |
| ecf55477df08 | 2026-07-15 | 1 | important | chore(tests): AUT-851 move NCCL defaults from run_ci_test.sh |
| 38626c288080 | 2026-07-15 | 1 | important | Pair frozen FSDP backward hooks (#5710) |
| 53a2dd56fec6 | 2026-07-15 | 1 | important | IMA fix by making the copy of book keeping buffer to GPU blo |
| 3927d9b96182 | 2026-07-15 | 1 | important | feat(docker): Add NCCL installation script and install NCCL  |
| 8bf73659f692 | 2026-07-15 | 1 | important | Set num_splits to 0 for FA4 inference (#5804) |
| ccdfa7dccd10 | 2026-07-15 | 1 | important | Missing moe_router_dtype causes unexpected downcast in Model |
| 4bf7fca050ae | 2026-07-14 | 1 | important | Fix MegatronFSDP root module hook dispatch (#5808) |
| 3c4645201af5 | 2026-07-14 | 1 | important | fix(clip_grads): handle empty grads_for_norm in inf-norm and |
| 33706f968e0e | 2026-07-14 | 2 | important | Implement Quantile Balancing in MoE (#5349) |
| a3279434e64b | 2026-07-14 | 2 | important | Exercise nested MFSDP CUDA graph capture (#5796) |
| e5344abbdf81 | 2026-07-14 | 1 | important | Mamba prefix caching fixes (#5502) |
| a79f49d37bd9 | 2026-07-14 | 1 | important | Delegate reasoning token retention to the chat template in m |
| bcf4c8fb5179 | 2026-07-14 | 1 | important | Inference: Add load aware routing to prefix caching.  (#5607 |
| 3a253ac5c2d3 | 2026-07-14 | 1 | important | Inference: Do not route pad/dummy tokens to any expert (#492 |
| a28ca480bb00 | 2026-07-14 | 1 | important | Short-circuit condition to avoid copying from GPU memory in  |
| eadbaa6189fb | 2026-07-13 | 1 | important | Inference: Add profile endpoints to chat completions.  (#561 |
| 2c579b45b96b | 2026-07-10 | 1 | suggestion | Set Bert TE spec q/k_layernorm to None (#5687) |
| e1b845433013 | 2026-06-25 | 1 | important | Fix fused MLA down projection with tensor parallelism (#5383 |

## 与 claude[bot] 实际评论对比
- clone 的 PR 元数据不含 review 评论正文，无法离线取得 claude[bot] 评论 → **跳过此对比并记录**（§6）。

## 价值说明
- 若 top-20 中有真问题，即『这套东西能用』的最直接证据，也是给 NVIDIA 的第一份材料。
- 本节为定性；未做 GPU 复现，finding 真伪待人工/CI 核验。
