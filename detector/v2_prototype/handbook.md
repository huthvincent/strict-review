# 叶子手册（迷你版·train 频次 top15·确定性构建 2026-07-19）
> 每页：该类问题的机制画像、症状、显现条件、历史高发文件。供检测 agent 常驻参考。

## suboptimal-kernel-gate（train 案例 247 起）
- 症状: throughput(184), latency(48)
- 常见显现条件: CUDA GPU; fp16/bf16; vLLM V1
- 历史高发文件: vllm/model_executor/layers/quantization/fp8.py, vllm/platforms/rocm.py, megatron/core/transformer/attention.py

## redundant-hot-path-work（train 案例 201 起）
- 症状: latency(149), throughput(39)
- 常见显现条件: CUDA GPU; CUDA; GPU
- 历史高发文件: vllm/v1/worker/gpu_model_runner.py, vllm/sequence.py, vllm/v1/core/sched/scheduler.py

## fused-backend-swap（train 案例 197 起）
- 症状: throughput(121), latency(71)
- 常见显现条件: CUDA GPU; NVIDIA GPU with Transformer Engine; NVIDIA SM100 / Blackwell (device capability 100)
- 典型机制: Bumps the pinned vllm-flash-attention GIT_TAG (93cf5a08... → 2d3b7508...) to pull in an upstream FA3 kernel improvement for the attention-sinks code path, speeding up FlashAttention-3 execution when attention sinks are used.
- 历史高发文件: vllm/envs.py, CMakeLists.txt, vllm/_custom_ops.py

## kernel-fusion（train 案例 193 起）
- 症状: throughput(96), latency(84)
- 常见显现条件: CUDA GPU; CUDA GPU (Triton kernels); GPU
- 历史高发文件: csrc/ops.h, csrc/torch_bindings.cpp, vllm/model_executor/layers/fused_moe/fused_moe.py

## kernel-config-tuning（train 案例 156 起）
- 症状: throughput(111), latency(30)
- 常见显现条件: fp8_w8a8; NVIDIA H100 80GB HBM3; NVIDIA H20-3e
- 历史高发文件: vllm/model_executor/layers/fused_moe/fused_moe.py, benchmarks/kernels/benchmark_moe.py, vllm/v1/attention/backends/pallas.py

## config-toggle-perf-feature（train 案例 149 起）
- 症状: throughput(80), latency(32)
- 常见显现条件: fp16; CUDA GPU; V1 (VLLM_USE_V1)
- 历史高发文件: vllm/envs.py, vllm/engine/arg_utils.py, vllm/config.py

## cudagraph-enablement（train 案例 112 起）
- 症状: gpu-util-or-bubble(73), latency(22)
- 常见显现条件: NVIDIA GPU (CUDA graphs); CUDA GPU; CUDA GPU (CUDA graphs)
- 历史高发文件: megatron/core/transformer/cuda_graphs.py, vllm/v1/worker/gpu_model_runner.py, megatron/core/transformer/transformer_layer.py

## blocking-d2h-sync（train 案例 98 起）
- 症状: latency(63), gpu-util-or-bubble(27)
- 常见显现条件: CUDA GPU; GPU (CUDA); vLLM V1
- 历史高发文件: vllm/v1/worker/gpu_model_runner.py, vllm/model_executor/models/qwen2_5_vl.py, vllm/model_executor/models/qwen2_vl.py

## redundant-buffer-copy（train 案例 93 起）
- 症状: memory(43), latency(43)
- 常见显现条件: CUDA GPU; NVIDIA GPU (CUDA graphs); CUDA
- 历史高发文件: vllm/v1/worker/gpu_model_runner.py, vllm/worker/model_runner.py, csrc/ops.h

## missing-comm-overlap（train 案例 93 起）
- 症状: gpu-util-or-bubble(79), throughput(9)
- 常见显现条件: tensor-parallel with tp_comm_overlap (Userbuffers) enabled; MoE models with shared experts; golden values updated for DGX H100/A100
- 历史高发文件: megatron/core/pipeline_parallel/schedules.py, megatron/core/model_parallel_config.py, megatron/arguments.py

## boolean-guard-misfire（train 案例 91 起）
- 症状: throughput(41), latency(15)
- 常见显现条件: CUDA GPU; vLLM V1; GPU (CUDA graphs)
- 历史高发文件: vllm/engine/arg_utils.py, vllm/model_executor/layers/fused_moe/layer.py, megatron/core/pipeline_parallel/schedules.py

## buffer-sizing-layout（train 案例 80 起）
- 症状: memory(45), throughput(19)
- 常见显现条件: CUDA GPU; int32; vLLM V1 engine
- 历史高发文件: vllm/envs.py, vllm/v1/worker/gpu_model_runner.py, vllm/v1/attention/backends/pallas.py

## memory-leak-unpruned-state（train 案例 77 起）
- 症状: memory(72), latency(2)
- 常见显现条件: CUDA GPU; fp8; fp16
- 历史高发文件: megatron/model/transformer.py, megatron/schedules.py, vllm/envs.py

## collective-payload-reduction（train 案例 75 起）
- 症状: throughput(57), latency(12)
- 常见显现条件: CUDA (torch.cuda.FloatTensor); Transformer Engine; worse with large common state dicts / large world_size
- 历史高发文件: megatron/training/arguments.py, vllm/distributed/device_communicators/cuda_communicator.py, megatron/core/transformer/transformer_config.py

## redundant-collective（train 案例 68 起）
- 症状: throughput(37), latency(18)
- 常见显现条件: multi-GPU; multi-GPU/multi-node; tensor-parallel (TP>1); sequence-parallel path also touched
- 历史高发文件: megatron/training/arguments.py, megatron/core/dist_checkpointing/strategies/fully_parallel.py, megatron/training/checkpointing.py
