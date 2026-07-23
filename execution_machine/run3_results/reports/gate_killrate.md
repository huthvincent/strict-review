# 画像门 train 误杀率 (RUN3 §3.1)

- generated: 2026-07-23 · 在全部 3386 个 train 正样本上跑排除规则
- **误杀率 = 3/3386 = 0.09%**（阈值 ≤2%）: ✓ 通过
- 门规则: 改动文件全部落在 tests|docs|examples|.github 且不触及 recipe/config 路径 才 no_issue

## 被误杀样本清单（全部为 examples/ 中的边界案例，机制真在示例代码里）

- `Megatron-LM@dfb78dca320d` (leaf: mismatched-collective-config) — 改动: examples/run_simple_mcore_train_loop.py
- `Megatron-LM@94dbfd1cd35f` (leaf: stream-serialization) — 改动: examples/pretrain_bert_distributed.sh, examples/pretrain_bert_distributed_with_mp.sh, examples/pretrain_gpt_distributed.sh
- `vllm@8820821b5912` (leaf: async-output-overlap) — 改动: examples/lmcache/disagg_prefill_lmcache_v1/disagg_proxy_server.py

结论：3 例均为 examples/ 目录内被标为 perf 回归的边界案例，门排除 examples/ 是正确设计；0.09%% 远低于阈值，规则无需收窄。
