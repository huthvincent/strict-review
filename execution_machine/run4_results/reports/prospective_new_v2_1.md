# RUN4 v3 §4.2 — 新前瞻窗（22 未见 commit，真前瞻）

- generated: 2026-07-23T07:00:10.254345+00:00 · 冻结 v2.1 · 窗口 2026-07-20..(fetch 2026-07-23) · leak=0
- **触发率（severe）: 11/22** · 被门谓词命中 6
- 这是数据集与 v2.1 设计窗口**之后**的新提交 → 真前瞻（区别于 §4.1 诊断回放）。

## 触发 commit（severe，按 conf_raw）

### `5a88c57bdcd6` (conf_raw 0.75) — Support HSDP deferred DP-outer gradient reduction (#5743)
- important/route: 该改动重写 FSDP experimental 的梯度归约路径（reduce_partial_gradients），把 HSDP 的 DP-outer 归约从每 microbatch 改为延迟到最后一个 microbatch 一次性 finalize，直接触及每 backward 的通信热路径（reduce-scatter / all-r

### `8d16c6729541` (conf_raw 0.62) — [Main] Numerical fix for moe single grouped weight with fp8 fp4 primary wei
- important/route: 该改动触及 low-precision-state / DDP 参数同步类风险：修改了 fp8/fp4 grouped 量化权重在 distributed_data_parallel 与 param_and_grad_buffer 的存储重映射与 param all-gather 后处理（_post_param_sync），属于分布式优化

### `f41ec54958a0` (conf_raw 0.62) — Overlap async scheduling phases (#5549)
- important/route: 该改动重构了动态推理异步调度的书记(bookkeeping)H2D 发布路径：将原本内联在 initialize_attention_state 的立即发布拆分为可延迟发布 + 事件跟踪的 non_blocking 拷贝，并新增 copy_async_sched_sample_to_forward / commit_sampled_tok

### `cc5b09239c3f` (conf_raw 0.60) — Reduce MimoOptimizer update-success across the world for cross-grid consens
- important/route: 该改动在 MimoOptimizer.step() 热路径新增一次 world 范围 all_reduce(MIN) 做 update-success 共识，触及通信/调度性能面（每个 optimizer step 一次跨世界集合通信），建议验证其开销与是否可合并进已有的 found_inf 归约。

### `e4ea85970b26` (conf_raw 0.55) — [Main] Numerical fix for FC2 expert bias scales when using `use_transformer
- important/route: 改动修改了 TEGroupedMLP 的 op-fuser 融合 MoE experts 前向路径（每步每 expert 的核心 GEMM 热路径）：新增 _is_fused_impl_supported 门控——当 FC2 带 bias 且安装的 TE GroupedLinear 缺少 scale_bias 参数时返回 False，从而

### `890247c0944d` (conf_raw 0.55) — fix(resharding): stabilize NVSHMEM refit copy service (#5915)
- important/route: 改动触及 NVSHMEM resharding refit copy service 的执行/通信路径（pipeline_executor.execute_pipeline 每迭代循环、service init/schedule/run）及一个 NVSHMEM_MAX_CTAS 配置强制项，属于 reshard 通信吞吐面，需验证 ref

### `602fad039a96` (conf_raw 0.55) — Inference: Do not let prompt tokens return from the engine, unless requeste
- important/route: 该改动触及 IPC/传输开销类风险：修改了 DynamicInferenceRequest.serialize 及服务端 endpoints 的 engine->coordinator->API 序列化热路径，通过 return_prompt_tokens 默认丢弃 prompt_tokens 张量以降低长 prompt 的 ZMQ 传输

### `12c05a2d8979` (conf_raw 0.55) — Add compatibility between training CGs and CP>1 (#5894)
- important/route: 该改动为 PackedSeqParams 新增 pad_between_seqs 字段并将其传递给 TEDotProductAttention（thd/packed 注意力 + CP>1 + 训练 CUDA graph 捕获路径），触及计算/注意力热路径与 CUDA graph 捕获兼容性，需验证捕获/回放不退化。建议验证。

### `337c061c9c9a` (conf_raw 0.55) — Add fully_shard_optimizer for mixed-precision FSDP (#5411)
- important/route: 该改动新增 fully_shard_optimizer，在优化器 step 前后挂钩子，对 mixed-precision FSDP 分片参数按 grad.dtype!=param.dtype 逐参数把梯度 cast 到参数精度（通常 bf16→fp32），触及优化器 step 热路径（额外 cast 计算+临时高精度梯度缓冲）与低精度状

### `cfb116f28875` (conf_raw 0.45) — Add NeMo waveform audio processor (data-side feature extractor) (#5570)
- important/route: 新增 NeMo 音频波形处理器（data-side 特征提取：波形解码/拼接/log-mel materialization）触及数据预处理性能面。多模态 per-request CPU 预处理若由许多小 torch 算子构成、或每样本做 decode→concat→log-mel，一旦接入 dataloader 热路径可能成为吞吐瓶颈。

### `bd0872dda07a` (conf_raw 0.40) — Reduce boilerplate around MultiStorageClient feature checks (#5269)
- important/route: 该改动重构了 MSC 特性检查（indexed_dataset 的 _FileBinReader.read / _MMapBinReader 等数据读取路径每样本被调用），触及数据加载热路径；若新的 open_file/helper 在每次读取时引入额外的 is_enabled()/import_package() 调用或改变默认分支（走

