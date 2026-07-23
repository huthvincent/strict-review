# RUN4 v3 Stage 4 — 诊断性回放 v2.1（**非前瞻**）

- generated: 2026-07-23T06:54:42.727435+00:00 · 冻结 v2.1 · 窗口 2026-06-15..2026-07-19 · leak=0
- ⚠️ **v2.1 配方设计曾参考本窗口失败案例（尸检引用过 bcf4c8fb）；本节仅作诊断对照，
  禁止写成前瞻优势主张。** 真前瞻需新窗（见 §4.2）。
- **触发率（severe 口径）**：含门 **28/61** · 不含门(排除 gate_predicate) 28/61 · 被门谓词命中 15
- 对照：v1 severe 24/61（RUN2 发布口径 26/61 差异因当时含非-severe 计法）· v2.0 0/61
- 交并差 vs v1：交集 20 · v2.1 独有 8 · v1 独有 4

## v2.1 独有触发 top10（severe，标注门影响）

### `740c16e6b80a` (conf_raw 0.65) — Inference: Extend default cuda-graph coverage to 512 tokens (#5797)
- important/route: 该改动扩大了 prefill/mixed CUDA graph 的默认捕获覆盖范围（新增 cuda_graph_max_tokens=512，默认从原来的 decode bound `max_requests*(num_speculative_tokens+1)` 提高到 512 tokens），直接改变了推理热路径的 CUDA graph 捕获数量/覆盖率
- suggested_benchmark: 动态批推理端到端吞吐/延迟 + 捕获期显存峰值对比：默认 cuda_graph_max_tokens=512 vs 旧行为（=decode bound）

### `ffbe018c8e46` (conf_raw 0.55) — [refactor] Common combined-1F1B schedule-plan base (1/4 of #4798) (#4941)
- important/route: 该改动重构了 combined-1F1B 细粒度调度 plan 的 layer-callable 构建代码（build_layer_callables / MTP builder），这些 callable 决定了 MoE dispatch/combine 通信与 attention/MLP 计算在不同 CUDA stream 上的重叠划分，直接位于每 mic
- suggested_benchmark: MoE(EP>1)+combined-1F1B overlap 开启下的 per-iteration step time 与 A2A 重叠率（nsys）对比 parent

### `97b25d9d7b86` (conf_raw 0.55) — build: Update Transformer Engine to 2.17 (#5680)
- important/route: 该改动将 Transformer Engine 依赖从 2.16 升级到 2.17（pyproject.toml/uv.lock/Dockerfile），TE 是核心计算后端，提供 fused attention、GEMM、fused RoPE/layernorm、FP8 等热路径 kernel；版本切换会改变 kernel 选择、autotuning 与数
- suggested_benchmark: 对典型 GPT 训练配置（含 FP8/fused attention/fused RoPE）跑 TE2.16 vs TE2.17 的 step-time、tokens/s、峰值显存对比；覆盖 attention/GEMM/layernorm 关键 kernel 的 nsys profile 对比

### `834abc195025` (conf_raw 0.55) — Pin cudnn-fe and cuTeDSL version (#5812)
- important/route: 该改动修改了驱动融合 DSA(sparse attention) 计算路径的内核依赖版本（新增 nvidia-cudnn-frontend[cutedsl]==1.26.0，并把 nvidia-cutlass-dsl 4.6.0→4.5.0、quack-kernels 0.6.1→0.4.1），这些库直接支撑 dsa_cudnn_kernels.py 的融合
- suggested_benchmark: 在 dsa_kernel_backend='cudnn' 配置下跑 DSA attention 前/后向 microbenchmark，对比 1.26.0/4.5.0/quack-0.4.1 与旧组合的吞吐与显存

### `2c579b45b96b` (conf_raw 0.55) — Set Bert TE spec q/k_layernorm to None (#5687)
- important/route: 该改动将 BERT TE spec 的 q_layernorm/k_layernorm 由 IdentityOp 改为 None，触及模型构建/计算图配置面：当 config.qk_layernorm=True 时，SelfAttention 的 fallback（`submodules.q_layernorm or TENorm`）会实例化真实的 TENo
- suggested_benchmark: BERT-TE 预训练 step time / peak memory 对比：qk_layernorm=True 时新旧 spec 各跑 N step，测 attention 前向/反向耗时与激活显存

### `e97ac83603e4` (conf_raw 0.50) — Various ModelOpt fixes: QAD test for CICD, use model builder config instead
- important/route: 该改动重构 ModelOpt 启用检测（maybe_enable_modelopt）并将 KD teacher checkpoint 加载移入独立函数 load_kd_teacher_checkpoint，触及模型构建/权重加载启动路径（redundant-load-startup-work 类风险）：maybe_enable_modelopt 现在在 ge
- suggested_benchmark: 对比 modelopt distill/QAD 恢复场景的启动时长（get_model + setup_model_and_optimizer + teacher 加载）与父快照差异

### `a00c0de8530b` (conf_raw 0.40) — Add MimoModel.zero_grad_buffer delegating to active DDP submodules (#5372)
- important/route: 新增 MimoModel.zero_grad_buffer 在每个训练迭代（grad-accum 周期）被调用，触及训练热路径的 DDP 梯度缓冲管理；实现仅对 active 子模块 fan-out 到既有 DDP.zero_grad_buffer，不引入额外同步或冗余工作，风险低，但属性能面改动，建议验证。
- suggested_benchmark: MimoModel 训练迭代计时：对比该方法调用前后单步 host 侧开销，确认无多余 buffer 遍历/同步

### `4abeecce745f` (conf_raw 0.40) — Fix averaging for MoE z-loss metric tracking (#3199)
- important/route: 该改动为 z_loss 指标新增 avg_group=tp_dp_cp_group，在 MoEMetricsTracker._sync_metrics 中会对该指标多触发一次 all_reduce(op=AVG)（over tp_dp_cp 组），触及分布式通信面，建议验证是否引入每步额外集合通信开销及是否与既有 needs_dp_avg 的 DP AVG 
- suggested_benchmark: MoE 训练端到端 step 时间对比（开/关 z_loss，含 TP+DP+CP）

