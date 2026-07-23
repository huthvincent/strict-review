# RUN3 Stage 5 — 前瞻对决 v2 vs v1（同一 61 commit）

- generated: 2026-07-23T00:48:09.832069+00:00 · 冻结 detector_v2 · budget=2 · leak 0 · 窗口 2026-06-15..2026-07-19
- **v2 触发率（含门）: 0/61 (0%)** · v2 被画像门直接排除: 15 个（单列一类）
- v1 触发（RUN2）: 24/61
- 交集 0 · v2 独有 0 · v1 独有 24

## v2 独有触发 top10（按 confidence）

## v1 独有触发 top10（v2 漏报，诊断 v2 过度沉默）

- `e5344abbdf81` (conf 0.75) — Mamba prefix caching fixes (#5502) — important: 此改动触及 prefix-state-caching（inference-serving），该类问题历史上主要在 [DCP/PCP asserted world_size==1 for hybrid/sliding-window paths
- `acc7e644572b` (conf 0.75) — fix bug where Gemma4 is not working with recompute_granularity = "full — important: 此改动触及 activation-recompute（memory-footprint），该类问题历史上主要在 [MoE (shared experts, TE grouped MLP), optionally MLA, context p
- `53a2dd56fec6` (conf 0.75) — IMA fix by making the copy of book keeping buffer to GPU blocking (#57 — important: 此改动触及 blocking-d2h-sync（concurrency-sync），该类问题历史上主要在 [inference dynamic batching (dynamic inference context), distribute
- `4a1f74350e76` (conf 0.75) — Avoid extra MFSDP v2 model-weight sync memcpy (#5834) — important: 此改动触及 redundant-buffer-copy（memory-management），该类问题历史上主要在 [TP (low latency TP case mentioned) (issue-measured), MoE/expe
- `bcf4c8fb5179` (conf 0.72) — Inference: Add load aware routing to prefix caching.  (#5607) — important: The default coordinator routing policy is changed from FIRST_PREFIX_BLOCK to LOAD_BALANCED in three places simultaneousl
- `a79f49d37bd9` (conf 0.72) — Delegate reasoning token retention to the chat template in multi-turn  — important: In megatron/rl/inference/megatron.py::base_generate (the per-request hot path for RL rollouts), the extra_body now uncon
- `a28ca480bb00` (conf 0.72) — Short-circuit condition to avoid copying from GPU memory in `ChainedOp — important: 此改动触及 blocking-d2h-sync（concurrency-sync），该类问题历史上主要在 [inference dynamic batching (dynamic inference context), distribute
- `8bf73659f692` (conf 0.72) — Set num_splits to 0 for FA4 inference (#5804) — important: 此改动触及 kernel-config-tuning（kernel-efficiency），该类问题历史上主要在 [TP8, TP4, MoE (OpenAI triton_kernels matmul_ogs)] 配置下显现；The ch
- `f8e1ac64b058` (conf 0.70) — Refactor data parallel coordinator to enable modular handlers (#5550) — important: static rule 'numpy-index-gpu-tensor-d2h-sync' matched antipattern: Indexing a GPU tensor with a NumPy array (or CPU-deri
- `f258d4fa82de` (conf 0.70) — Test zero-CTA copy-engine all-gather (#5858) — important: static rule 'full-cuda-sync-in-overlap-path' matched antipattern: Calling the global torch.cuda.synchronize() (full devi

## 结论（定性，无标签）
- v2 触发率 0% vs v1 39%：v2 显著更沉默，与 dev 过度抑制一致。
- 被画像门排除的 commit 单列见上；leak 纪律：全程 leak_attempt = 0。
