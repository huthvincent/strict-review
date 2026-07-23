# MegaPerfBench 检测手册 v1（74 叶，train-only 蒸馏）

- generated: 2026-07-22T23:52:41.003996+00:00 · model us.anthropic.claude-opus-4-8 · prompt handbook.v1 · 仅从 train 卡蒸馏
- 覆盖 73/74 叶（1 叶无 train 卡，如实跳过）

**跳过叶（无 train 卡）**: ['weight-transfer-sync']


## 大类：collective-comm

### collective-payload-reduction  （75 张 train 卡）

**典型反模式**：
- Collecting full per-rank state onto rank 0 via gather_object/all_gather_object when only rank 0 (or one consumer) reads the result — O(world_size × payload) peak memory/serialization instead of broadcast_object_list or a single-rank path.
- Allocating a global-batch (or full-vocab) zero tensor, writing only the local slice, then all_reduce-summing the entire padded tensor — reducing mostly-zero volume instead of an all-gather/scatter of just the local contribution.
- All-gathering hidden states / activations in full precision (bf16/fp16) before quantization, when the tensors could be quantized (fp8/fp4) prior to the collective to halve/quarter comm volume; likewise disabling fp8 param-gather and falling back to a bf16 all-gather.
- Replicating all tokens to every expert-parallel rank (AllGather dispatcher) rather than exchanging only the tokens each rank needs via all_to_all_single with per-rank split sizes; computing the collective over padded/full spans (full context logits, padded packed bins, full-TP-sized activation) instead of sliced real tokens/chunks.
- Issuing per-parameter/per-bucket collectives in a Python loop, or reducing over a wider communicator (TP+DP) than the value actually varies across (e.g. amax reduction over TP-DP instead of TP).

**显现条件 top3**：
- Distributed data-parallel / ZeRO-style optimizer or FSDP with param all-gather and grad reduce-scatter; comm volume scales with dtype width (fp32 grad buffer vs fp16/bf16/fp8 params) and world_size.
- Expert-parallel MoE dispatch/combine (all-to-all or all-gather) where payload precision and token routing determine volume; benefit grows with EP world size, high spatial redundancy, or quantizable activations.
- Pipeline+tensor parallelism P2P activation transfer, or checkpoint-save common-state-dict collectives, where the transferred payload is TP-redundant / world_size-replicated and could be chunked, quantized, or single-sourced.

**历史量级样本**：
- up to 12.5x per-item cache-space savings (100 placeholder tokens vs 8 real embeddings; removed *12.5 formatting multiplier)
- rank-0 peak memory O(world_size × common-state-dict size) under gather_object before switching to broadcast_object_list

**检测时该查什么（父快照核验）**：
- Is the collective on a hot path (per-step training/inference forward, per-bucket optimizer step) vs a rare setup/checkpoint path? A payload-reduction revert or precision fallback here regresses steady-state throughput.
- Was the payload previously reduced (sliced to real tokens, quantized before the collective, TP-chunked, single-rank broadcast) and did the change re-materialize the full/padded/full-precision tensor?
- Is the value actually varying across all ranks in the chosen communicator, or is it redundant (identical across TP ranks, zero outside local slice, consumed by only rank 0)? Over-wide group = wasted volume.
- Is the optimized path opt-in or default-enabled? Check whether a validate_args/config guard silently disables fp8 param-gather or a scatter-gather path, doubling comm volume by default.
- Does dtype/parallelism gating (dp_size>1, use_ep, TP>1 & SP off) match where the reduction applies — a mismatch leaves the full-volume fallback active.

**高发文件 top5**：`megatron/core/distributed/**/*fsdp*.py (custom/Megatron FSDP param-gather, reduce-scatter, aggregate collectives)`, `megatron/core/optimizer/distrib_optimizer.py (ZeRO-style reduce_scatter / all_gather over grad & param buffers)`, `megatron/core/pipeline_parallel/p2p_communication.py (_communicate, scatter_gather_tensors_in_pipeline)`, `megatron/core/dist_checkpointing/**/*.py (common-state-dict consistency gather/broadcast_object_list)`, `vllm/model_executor/layers/fused_moe/**/*.py (DeepEP/flashinfer dispatch-combine, post-quant all-gather, all_to_all_single)`

### ipc-transport-overhead  （44 张 train 卡）

**典型反模式**：
- Serializing whole objects (including large tensor/ndarray buffers) through pickle/msgpack default paths on the IPC hot path — e.g. `pickle.dumps(obj, HIGHEST_PROTOCOL)` or a msgpack Ext(CUSTOM_TYPE_PICKLE, pickle.dumps(...)) fallback — which forces a full memcpy of every buffer instead of using zero-copy (pickle protocol-5 buffer_callback / out-of-band backing buffers).
- Exchanging tensor shards between ranks via object-collective paths (all_gather_object / gather_object) that pickle and host-serialize device tensors, instead of a device-side tensor collective.
- Routing intra-node collectives (all_reduce/all_gather) through a generic distributed backend (Gloo/RCCL/NCCL fallback) rather than a specialized IPC/shared-memory or custom P2P kernel path, incurring higher latency for small messages.
- Hardcoding or defaulting transport-selection flags (env vars / config) so the slower transport is chosen (e.g. NVLink path disabled by default, TCP loopback socket instead of Unix-domain/IPC socket, object-collective vs zero-copy), or leaving the fast path opt-in so it never runs by default.
- Failing to null-out / short-circuit already-cached payloads (mm_hash cache hits) so large multimodal buffers get re-serialized and IPC-transferred every batch.

**显现条件 top3**：
- Multi-process/multi-rank single-node IPC on the request or RPC hot path (TP/PP local readers, V1 engine-core↔client over ZMQ, driver→worker broadcast).
- Large or repeated payloads at the boundary: multimodal tensors, long prompt text, or RPC objects exceeding the fast-path chunk size and falling back to a slower transport.
- Collective communication (TP all_reduce/all_gather, EP MoE all-to-all dispatch/combine, PP stage tensor passing) where transport choice — NVLink vs IB, IPC/SHM vs socket, custom kernel vs vendor NCCL/RCCL/Gloo — governs latency.

**检测时该查什么（父快照核验）**：
- Is this on a per-request / per-step hot path (engine↔client IPC, RPC broadcast, per-token collective)? If so, any full-object serialization is suspect.
- Does the serialization path copy tensor/ndarray buffers into the message (pickle/msgpack default) rather than passing them out-of-band / zero-copy?
- Which transport is selected by default? Check env-var/config defaults and hardcoded flags — is the faster path (NVLink, IPC/Unix socket, custom kernel, SHM) opt-in or disabled by default?
- For collectives: does the code fall back to a generic backend (Gloo/RCCL/object-collective) when a specialized intra-node/IPC/SHM path exists?
- For cached payloads: is there a guard that skips re-serializing/re-transferring items already present in the peer's cache (mm_hash)?
- Is there a size threshold where the fast path degrades to a slow fallback (chunk-size limit, max_size for custom allreduce)? Confirm typical payloads don't routinely exceed it.
- Check message/handle binary format assumptions across library versions (e.g. CUDA IPC handle size) that could silently break the fast path.

**高发文件 top5**：`vllm/distributed/device_communicators/shm_broadcast.py`, `vllm/distributed/parallel_state.py (send/recv_tensor_dict, device_communicator)`, `vllm/v1/serial_utils.py / MsgpackEncoder-Decoder (V1 engine IPC serialization)`, `csrc/custom_all_reduce.cu / csrc/cpu/shm.cpp (custom IPC/SHM collective kernels)`, `megatron/core/ (nccl_allocator.py, dist_checkpointing exchange_loaded_tensors, symmetric-memory / multimem collective layers)`

### mismatched-collective-config  （55 张 train 卡）

**典型反模式**：
- Deriving distributed rank/world_size from per-node values (e.g. LOCAL_RANK, torch.cuda.device_count()) instead of global RANK/WORLD_SIZE, so multi-node processes register colliding ranks and a too-small world size.
- Guarding a collective call (all_reduce/all_gather/reduce_scatter/broadcast) inside a rank-local data-dependent condition like `if len(local_params) > 0:` — ranks that skip the branch never enter the collective and the group deadlocks.
- Issuing collectives while iterating a non-deterministic / per-rank-ordered container (dict/set of params, `sorted(timers.items())` with rank-varying keys), so collective launch order or the set of participating keys diverges across ranks.
- Calling all_gather/all_reduce on tensors whose shape or dtype is data-dependent per rank (variable first-dim, AMP fp32-vs-half up-cast on the P2P/pipeline path) without padding or shape-agreement, violating NCCL's identical-shape/dtype requirement.
- Choosing the collective process group incorrectly: defaulting a reduction to the whole world (missing fp8_group / amax_reduction_group), using wrong src rank for a group broadcast, or wrong with_context_parallel scoping — right op, wrong communicator.
- Returning early (EMPTY output on num_scheduled_tokens==0) or short-circuiting a per-step loop on one DP rank so it skips a collective the other ranks still execute.

**显现条件 top3**：
- Multi-node / multi-GPU NCCL runs where global world size exceeds a single node's GPU count, or DP world size is large — divergence only appears once >1 group member exists.
- Uneven per-rank workloads: distributed optimizer with unevenly sharded main-params, MoE+Dense hybrids with uneven expert-layer distribution across pipeline stages, or variable-length per-rank inputs (e.g. neg-context tokens).
- Advanced parallelism combos: TP+PP, expert/context parallelism, FSDP, DP+EP with microbatching/DBO, and FP8/amax reductions where the effective process group is easy to mis-scope.

**历史量级样本**：
- MoE Parallel Folding: 64 GPUs → 8 GPUs minimum for CP=8,EP=8 (README minimal-GPU claim, not a throughput number)
- Symmetric NCCL memory registration: communication SM usage reduced from ~16-32 SMs to ~1-6 SMs
- Megatron-FSDP: up to 25% speed up and 23% memory savings (general FSDP README figure, not isolated to symmetric registration)

**检测时该查什么（父快照核验）**：
- Is every collective (all_reduce/all_gather/reduce_scatter/broadcast) reached unconditionally by ALL ranks in its group, or is it nested under a rank-local, data-dependent branch (`if len(...) > 0`, num_tokens==0, has-expert-params)?
- Are rank and world_size derived from GLOBAL env (RANK/WORLD_SIZE) rather than per-node LOCAL_RANK / device_count()?
- Do the tensor shape AND dtype fed to the collective match across ranks (check variable first-dims, AMP/mixed-precision up-casts on P2P/pipeline sends, padding of buckets)?
- Is collective launch order deterministic across ranks (no iteration over unordered dict/set, no rank-varying key sets in `sorted(timers.items())`)?
- Is the collective issued on the intended process group with the correct src rank and correct with_context_parallel / fp8 / amax group scope, not defaulting to the whole world?
- For per-step loops (inference DP, DBO/microbatch cudagraph dispatch): can any rank return early or take a different num-tokens path and skip a collective the others still run?

**高发文件 top5**：`megatron/core/parallel_state.py (initialize_model_parallel, group construction, get_nccl_options)`, `megatron/training/utils.py (calc_params_l2_norm and other collective-emitting reductions)`, `megatron/core/distributed/** (DDP grad bucketing, FSDP reduce-scatter, param/grad reduce pipelines)`, `megatron/core/dist_checkpointing/** (distributed checkpoint load exchange, per-key timer collectives)`, `vllm/v1/worker/gpu_model_runner.py (DP/EP coordinate_batch_across_dp, ubatch/DBO synchronize paths)`

### missing-comm-overlap  （93 张 train 卡）

**典型反模式**：
- Issuing collective ops synchronously (async_op absent/False) on the critical path — e.g. distributed-optimizer param all-gather (`_all_gather_base`/`gather_model_params`) or checkpoint-load `all_gather` runs, then blocks compute — instead of launching async and interleaving with the following compute region.
- Assertions/version-guards that hard-disable an overlap path for a config combination (e.g. `assert overlap not supported for MXFP8 param AG`, `overlap_grad_reduce forbidden for interleaved schedule`, MLA linears missing `tp_comm_buffer_name` triggering `assert ub_name is not None`) — silently forcing collectives back onto the serial path.
- Reading overlap-enabling config off the wrong object (e.g. checking `self.config` on a wrapping ChainedOptimizer instead of each chained sub-optimizer), so the pre-hook that launches async param all-gather never fires and the gather serializes with forward.
- Reordering compute so a communication-hiding window is destroyed — e.g. running shared-expert compute before the router/dispatch instead of interleaving it with the routed-expert A2A dispatch/combine, leaving the collective un-overlapped.
- Env/queue config that serializes launches (CUDA_DEVICE_MAX_CONNECTIONS=1) required for TP/SP ordering but which prevents FSDP param all-gather / reduce-scatter from overlapping with compute.
- Not forwarding buffer names / new TE overlap kwargs (`ub_name`, `ub_overlap_ag/rs`, `ub_overlap_rs_dgrad`, quantization_modes) through the module stack, so TE falls back to non-overlapped collectives on the affected version threshold.

**显现条件 top3**：
- Tensor-parallel + sequence-parallel with tp_comm_overlap (TE Userbuffers) enabled, on a specific TransformerEngine version threshold where the overlap kwarg wiring changes (e.g. >1.0.0, >1.5.0, >=1.10.0, >=2.7.0).
- Distributed optimizer + data-parallel with overlap_param_gather / overlap_grad_reduce (including ChainedOptimizer for MoE), where the param all-gather / grad reduce-scatter should overlap forward/backward compute.
- MoE Expert-Parallel (EP>1) All-to-All dispatch/combine, or interleaved/virtual pipeline (VPP) + P2P comm, where fine-grained scheduling is required to hide the collective behind compute.

**检测时该查什么（父快照核验）**：
- Is the collective on the hot path (per-iteration forward/backward, optimizer step, or per-microbatch P2P)? A synchronous collective here directly serializes with compute.
- Is async_op=True actually set AND is the returned handle waited on only after a compute region that can overlap it (not immediately)?
- At the parent snapshot, is overlap default-enabled or opt-in? Check argparse defaults (store_true vs store_false/--no-*) and whether a version guard / assertion disables it for the current TE version, dtype (fp8/mxfp8), or model variant (MLA, MoE shared experts).
- Is the overlap-enabling config read from the correct object (per sub-optimizer vs wrapper) and forwarded all the way down to the TE module (ub_name / ub_overlap_* kwargs plumbed through attention.py/mlp.py/transformer_engine.py)?
- Does an env/queue setting (CUDA_DEVICE_MAX_CONNECTIONS=1) or a compute reorder (shared-expert before router) collapse the window in which the collective was meant to be hidden?
- Are golden/timing tests updated so a silent loss of overlap (throughput/iteration_timing regression) would actually be caught?

**高发文件 top5**：`megatron/core/transformer/custom_layers/transformer_engine.py (TELinear/TELayerNormColumnParallelLinear ub_* kwarg wiring)`, `megatron/core/optimizer/distrib_optimizer.py + optimizer.py (param all-gather / grad reduce-scatter overlap, ChainedOptimizer pre-hooks)`, `megatron/core/pipeline_parallel/schedules.py + model_chunk_schedule_plan.py / combined_1f1b.py / fine_grained_callables.py (1F1B, P2P overlap, EP A2A fine-grained scheduling)`, `megatron/core/transformer/{attention.py,mlp.py,moe/*} (tp_comm_buffer_name assignment, MoE dispatch/combine ordering)`, `megatron/core/dist_checkpointing/strategies/fully_parallel.py (exchange_loaded_tensors all_gather async_op)`

### rank-placement-locality  （16 张 train 卡）

**典型反模式**：
- Constructing parallel rank groups from raw enumeration/allocation order (e.g. Ray-returned worker list) without sorting by node/host IP, so global ranks get interleaved across physical nodes and TP/EP groups span slow inter-node links instead of intra-node NVLink.
- Hardcoding a bare device target (`self.to('cuda')`, allocations/NCCL buffers without an explicit device) so every rank resolves to cuda:0 — piling context/buffers on GPU 0 and defeating per-rank placement.
- Choosing or reordering the rank-layout `order` string (e.g. tp-cp-ep-pp-dp vs tp-ep-pp-cp-dp) without accounting for which parallel group's stride crosses node boundaries, silently moving a latency-sensitive collective off the fast interconnect.
- Requesting fractional GPUs for workers (num_gpus=gpu_memory_utilization) so the scheduler packs multiple ranks onto one physical GPU, or setting the device-control env var too late in actor __init__ so device pinning never takes effect.

**显现条件 top3**：
- Multi-node clusters where GPU-to-rank mapping decides whether a collective runs over intra-node NVLink vs inter-node network — TP/EP/CP groups that cross node boundaries pay the penalty.
- Data-parallel or expert-parallel with all2all/DeepEP on Ray/MP backends, where DP-rank placement strategy (pack/fill/strict) and device isolation govern locality.
- Hierarchical context/expert parallelism (Ulysses A2A low-level + Ring P2P high-level) where sub-group construction must map fast links to low-level and slow links to high-level groups.

**历史量级样本**：
- ~3x allreduce speedup for 32MB message size on a 2-socket CPU machine (CPU SHM inference_all_reduce)

**检测时该查什么（父快照核验）**：
- Is the rank-group construction order derived from a topology-aware sort (driver-first, same-node contiguous, host-IP grouped) or from raw allocator enumeration order?
- Does any device selection use a bare 'cuda' / implicit cuda:0 rather than the process's LOCAL_RANK, especially in module __init__, zero.Init, or NCCL working-buffer allocation?
- For Ray/MP backends: are workers requesting a full GPU (not a fraction), and is the device-control env var set early enough to affect runtime init (before dpctl/CUDA context creation)?
- Is the parallel-layout `order` string / interleave knob default-changed on a hot collective path, and does the affected group's stride cross node boundaries under the target parallelism config?
- Under DP>1: are distinct physical GPUs visible to NCCL (not all masked to device 0 via CUDA_VISIBLE_DEVICES) so peer/P2P detection works?

**高发文件 top5**：`megatron/core/parallel_state.py, megatron/training/initialize.py (RankGenerator / order string / generate_masked_orthogonal_rank_groups)`, `vllm/executor/ray_*.py (Ray executor worker ordering, sort_by_driver_then_worker_ip, num_gpus requests)`, `vllm/v1/engine/core*.py & platform device-control env-var setup (DP CUDA_VISIBLE_DEVICES, DCP interleave config)`, `deepspeed/runtime/zero/ (zero.Init device set, MiCS shard/replicate groups) and deepspeed/runtime/pipe/module.py (PipelineModule device placement)`, `transformer_engine context-parallel attention comm (flash_attn_a2a_communicate / reorder_seq_chunks, hierarchical CP group creation)`

### redundant-collective  （68 张 train 卡）

**典型反模式**：
- Issuing a collective (all_reduce/all_gather/reduce_scatter) unconditionally when the runtime configuration makes it a no-op or redundant, e.g. not guarding a group communication behind `if sequence_parallel or expert_parallel_size > 1`, so it degenerates into a world-size-spanning collective even in the single-partition case.
- Performing a reporting/aux-loss reduction inside a per-microbatch function (loss_func) rather than once per optimizer step, so the collective count scales with the gradient-accumulation / microbatch multiplier instead of being amortized.
- Emitting one small collective per shard/param/bucket in a loop (per-parameter all_gather, per-shard broadcast, per-bucket reduce_scatter) instead of coalescing them into a single batched call via `_coalescing_manager` or a single all_gather over the full param list.
- Calling `torch.distributed.barrier()` immediately before another routine that itself begins with a barrier, or running the same metadata-exchange / integrity-validation collective on every save/load instead of using a cached-structure fast path guarded by a constant-structure assumption flag.
- Running gradient-sync / param-gather collectives that the current pass does not need: e.g. DDP per-backward all_reduce on every microbatch instead of using no_sync() until the last one, launching grad_sync_func in forward_only/inference epilogues, or unconditionally prefetching the next bucket's param all-gather.
- Replicating a weight across TP ranks (parallel_mode='duplicated') so each rank redundantly computes the same GEMM, or an extra AVG all_reduce added on top of the intended reduce over the same/overlapping group.

**显现条件 top3**：
- Multi-rank distributed run (world_size > 1) with TP/DP/CP/EP or pipeline parallelism enabled, where the collective spans a process group that is trivial or unnecessary in the active configuration (e.g. no sequence-parallel, expert_parallel_size==1, untied embeddings, forward-only pass).
- Gradient accumulation / many microbatches per step (DDP or pipeline schedules) where a per-microbatch collective is issued instead of a single per-step one — cost scales with microbatch count.
- Distributed-optimizer / custom-FSDP param-gather and grad reduce-scatter, or distributed-checkpoint save/load, where many small per-param/per-shard/per-bucket collectives are launched uncoalesced, or the same planning/metadata-exchange collective runs on every iteration without a cached-structure fast path.

**检测时该查什么（父快照核验）**：
- Is this collective issued on a hot path (per-microbatch loss_func, per-parameter/per-shard loop, per-backward DDP hook)? Could it be hoisted to per-step or coalesced into one batched call?
- Is the collective guarded by the config that actually requires it? Check for missing `if sequence_parallel / expert_parallel_size>1 / world_size>1 / not disable_grad_reduce / not forward_only` guards before all_reduce/all_gather/reduce_scatter.
- Does the process group degenerate (single rank, replicated weight, untied embeddings) so the collective is a no-op or pure overhead?
- Does the caller already synchronize? Look for back-to-back barriers, or a routine that internally barriers being preceded by an explicit barrier.
- For checkpoint/optimizer paths: is there a cached-structure / constant-structure fast path, and is the skip flag actually threaded through to the collective call site (not just computed and dropped)?
- For prefetch/overlap collectives: is next-bucket param_sync started unconditionally regardless of execution/registration order or pass type?

**高发文件 top5**：`megatron/core/transformer/moe/** (SwitchMLP/SharedExpertMLP forward, expert-parallel collectives)`, `megatron/core/tensor_parallel/** (ColumnParallelLinear, layers, cross_entropy, mappings/regions)`, `megatron/core/dist_checkpointing/** (fully-parallel save/load strategy wrappers, exchange/plan/metadata)`, `megatron/core/distributed/** (custom FSDP, distributed optimizer param-gather, grad bucket groups)`, `megatron/core/pipeline_parallel/schedules.py (1F1B grad-sync / no_sync epilogues)`


## 大类：compilation

### compile-cache-misconfig  （25 张 train 卡）

**典型反模式**：
- Cache-key omits a semantically-relevant factor: hashing only an allow-list of config fields (opt-in factors.append) or excluding env vars that actually change codegen, so a graph compiled under one setting is silently reused under another (e.g. partitioning on/off, config snapshot deltas). Prefer opt-out 'hash everything except a small ignore set'.
- Using an unhashable or identity-hashed object as a cache key: passing CUDA Tensors (hashed by object identity) into an @lru_cache-decorated function, or memoizing only with a small in-process @lru_cache(maxsize=N) for keys whose distinct count exceeds N — every distinct/new object misses and re-runs the expensive compile.
- Divergent cache-key computation between two code paths that must agree: one path (e.g. the normal compile flow) migrated to a new hashing scheme while a parallel path (e.g. AOT/serialize path) still calls the old/removed helper, producing mismatched keys and cache misses or stale reuse.
- Unstable / order- or count-dependent cache path or key: per-model cache dirs suffixed by a process-global counter, or arch lists left unsorted so the build command hash differs run-to-run, defeating the persistent cache.
- Shared persistent cache path across processes/ranks without per-process isolation: multiple TP/MP workers or ranks writing the same Triton/XLA/tuned-GEMM cache dir → races, or opened readonly so non-driver ranks can never populate their (slightly different) graphs.
- Config default flip or version-gated default (e.g. AOT-compile / standalone-compile enabled on newer torch) that silently switches the caching/codegen path and interacts badly with a disable-cache flag or missing deserialize argument, so artifacts are recompiled or fail to load.

**检测时该查什么（父快照核验）**：
- Is the value used as a cache key hashable by value? Watch for Tensors, wrappers, or mutable objects flowing into @lru_cache or torch.compile cache-key computation (identity hashing => guaranteed misses).
- Does the cache key include every factor that changes codegen/behavior? Check for opt-in allow-lists that can forget a field, excluded env vars, and version-gated features (graph partitioning, standalone compile, AOT) not folded into the key.
- Do all code paths that read/write this cache compute the key the same way? Grep for a second/older hash helper (AOT vs normal path) after a caching refactor.
- Is the cache path/key stable across runs and processes? Look for process-global counter suffixes, unsorted arch/config lists, and shared paths lacking a per-pid / per-rank component.
- Is the @lru_cache maxsize large enough for the distinct-key cardinality, and is the memo persistent when the work (FSM/graph compile) is expensive?
- At a default-change or version-bump snapshot: is a caching path newly enabled by default, and does it interact with a disable-cache flag, a missing deserialize argument (f_globals), or a readonly path?

**高发文件 top5**：`vllm/compilation/backends.py`, `vllm/compilation/caching.py`, `vllm/envs.py (compile_factors / hash_factors / env-var defaults)`, `vllm/compilation/*.py (InductorAdaptor, PostGradPassManager, CustomCacheManager)`, `vllm/worker/tpu_worker.py & TPU/XLA cache-init paths (xr.initialize_cache)`

### cudagraph-enablement  （112 张 train 卡）

**典型反模式**：
- Gating CUDA-graph capture/replay solely on a decode-only predicate (e.g. asserting `is_decode_only()` or keying static tensors on `*_decode_only`), so prefill/mixed/chunked-prefill steps silently fall back to eager and lose graph coverage.
- Guard conditions that miss legitimate graphable modes: e.g. `if self.training and torch.is_grad_enabled()` excludes frozen layers (training=True, grad disabled) and routes them to the eager `else` branch; or `cuda_graph_scope != 'full_iteration'` unconditionally skipping graph-manager instantiation.
- Lazy/interleaved graph creation during the first fwd/bwd pass (create_cudagraphs() as a no-op stub) leaving graphs uncaptured or captured out of execution order, inflating memory-pool usage and leaving kernel-launch overhead in place.
- Replay path that only accepts `hidden_states` and asserts no other kwargs, so any config needing attention_mask/context/rotary_pos_emb (encoder-decoder, masked attention) cannot be graphed at all.
- Leaving a device-to-host sync in the hot path (MoE router variable per-expert token counts / cuda_sync_point, dynamic-shape MoE layers) or lazy FP8 transpose/columnwise cast 'in the next forward', both of which make the region uncapturable and force eager decode.
- Requiring a cudagraphable RNG tracker via a hard assertion, so inference paths using a no-op RNG tracker cannot enable CUDA graphs.

**显现条件 top3**：
- Autoregressive inference decode where per-step kernel-launch overhead dominates single-token latency.
- MoE/expert-parallel decode with dynamic per-expert token counts causing D2H sync and eager fallback.
- Prefill/mixed and full-iteration scopes left uncaptured despite enable_cuda_graph.

**检测时该查什么（父快照核验）**：
- Is the capture/replay predicate too narrow (decode-only, training-only, single scope string)? Check whether prefill/mixed, frozen-layer, and full_iteration cases fall through to eager.
- Are there D2H syncs, dynamic shapes, or lazy FP8 casts inside the region being captured (MoE router token counts, cuda_sync_point, columnwise/transpose FP8 done 'next forward')? These block capture.
- Does the replay signature accept all kwargs the model needs (attention_mask/context/rotary_pos_emb), or does it assert away non-hidden_states args?
- Are graphs created eagerly at a defined point in execution order, or lazily/interleaved via a no-op stub — check for memory-pool bloat and missed captures.
- Is the RNG tracker graph-safe (cudagraphable / graphsafe_get_state), and is there a hard assertion that would reject the inference no-op RNG tracker?
- For dynamic batching: are token counts padded/aligned to fixed shapes (rounded to tp_size, expert-capacity padding) so graph coverage is keyed correctly, and is fp8 correctly asserted supported/unsupported for the chosen scope?

**高发文件 top5**：`megatron/core/transformer/cuda_graphs.py (CudaGraphManager/CudaGraphRunner)`, `megatron/core/transformer/transformer_block.py and transformer_layer.py (capture/replay dispatch, _should_capture)`, `megatron/core/inference/**/dynamic_engine.py and static/dynamic inference context (batch-dimension builder, decode/prefill gating)`, `megatron/core/transformer/module.py (GraphableMegatronModule, cuda_graph_scope guards)`, `megatron/core/transformer/moe/** and mamba layer files (MoE router pad-for-cudagraph, MambaLayer CudaGraphManager)`

### cudagraph-incompatibility  （53 张 train 卡）

**典型反模式**：
- Capturing a CUDA graph over ops that force device-host synchronization or produce data-dependent output shapes (e.g. indexing with a CPU-side count tensor, masked_select/nonzero, dynamic token routing) — these silently break capture or fall back to eager, killing the expected speedup.
- Sizing/gating the CUDA graph on the wrong quantity: comparing request count against a token-count-based captured dimension, misclassifying zero-length padding requests as prefills, or setting a wrong num_warmup_microbatches so the captured schedule order diverges from replay.
- Indexing per-graph state (mempools, buffer pools, VP-stage buffers) with a *mutable global* runtime rank/stage instead of the static stage the layer belongs to, so overlapping schedules (VPP/interleaved) alias each other's captured buffers.
- Assuming a fixed layer output shape/type at capture time (bare Tensor vs tuple) or a single dispatched code path, while runtime dispatches a different attention/kernel path per batch (TRTLLM vs FA, chunked-local subclass inheriting parent cudagraph_support) — capture mismatches replay.
- Enabling a broader/default cudagraph mode (PIECEWISE->FULL, non-decode steps, larger capture-size list, nested inductor graph_partition) without excluding models/backends/features that cannot be captured, causing correctness breaks or lost throughput on the excluded-but-not-excluded path.

**显现条件 top3**：
- CUDA graphs enabled (enable_cuda_graph / --external-cuda-graph / cudagraph_mode=FULL or FULL_AND_PIECEWISE, not enforce_eager) combined with a feature that dispatches dynamically: speculative decoding, MoE/EP token dispatch, chunked/local or MLA attention, or runtime seq-len-dependent backend selection.
- Advanced parallelism interacting with capture: VPP/interleaved pipelining (per-stage buffer/mempool aliasing), PP=1 warmup-count edge cases, data-parallel + DBO (dual-batch overlap), and TP-aware capture step sizing.
- Specific dtype/quant + backend combos: fp8/nvfp4 quant, one-shot KV-scale calibration, torchao online quant reload in RLHF loops, and torch>=2.9 inductor graph_partition splitting compiled functions inside the cudagraph path.

**历史量级样本**：
- ~5.3% throughput / ~4.4% TTFT improvement attributed to a PIECEWISE-cudagraph MoE-split change (given up when reverted)

**检测时该查什么（父快照核验）**：
- Is this a hot inference/decode path where CUDA graph replay is the steady state? Check whether the change adds an op with a device-host sync (CPU tensor indexing) or data-dependent output size (masked_select/nonzero/dynamic routing) inside the captured region.
- At capture time, does the graph gate/size on the same quantity used at replay? Verify request-count vs token-count, padding/zero-length requests, warmup-microbatch counts, and capture-size lists against the backend's actual max supported size.
- Does per-graph state get indexed by a static stage/layer identity, or by a mutable global (current VP rank, current stream)? Confirm current_stream()/mempool/buffer selection matches the stream the collectives/allocations actually use.
- Is the layer/backend output shape and dispatched code path invariant across all captured sizes? Check subclasses that inherit cudagraph_support, tuple-vs-Tensor outputs, and per-batch backend switches (TRTLLM/FA/MLA, seq-len thresholds).
- Is a default cudagraph mode/scope being broadened? Confirm every incompatible model/feature (encoder-decoder, pooling, deepseek_v32 MTP, one-shot calibration, high-throughput all2all decode) is explicitly excluded or falls back cleanly.

**高发文件 top5**：`vllm/v1/attention/backends/**/*.py (attention builders, cudagraph_support, build_for_cudagraph_capture)`, `vllm/v1/worker/gpu_model_runner.py & cudagraph mode init/_check_and_update_cudagraph_mode`, `vllm/config/**/*.py & VerifyAndUpdateConfig / SpeculativeConfig (cudagraph mode & capture-size defaults, enforce_eager)`, `megatron/core/transformer/cuda_graphs.py & CudaGraphManager (mempools, buffer reuse, capture)`, `megatron/core/**/token_dispatcher & moe/*.py, multi_token_prediction.py, pipeline_parallel schedules (make_graphed_callables, warmup microbatches)`

### dynamo-graph-break  （41 张 train 卡）

**典型反模式**：
- Calling into third-party/native kernels (aiter, flash_attn triton `apply_rotary`, deepgemm, punica bgmv/sgmv) directly as plain Python staticmethods/functions inside the traced region — Dynamo cannot trace into them (hits unsupported `hasattr`, data-dependent branching, or inline C++/ASM), producing a graph break. Wrap them in `direct_register_custom_op` / `torch.library.custom_op` with a `register_fake`/meta implementation.
- Reading concrete Python-int tensor shapes (`m, _ = x.shape` / `w_n, _ = layer.weight.shape`, or calling `.item()`/`hasattr`/device-capability oracles like `from_oracle()`, `current_platform.is_device_capability()`) on the hot forward path, forcing shape specialization → graph break + recompilation. Use symbolic indexing (`out.shape[0]`) and hoist capability checks out of the traced region.
- Function-local (inside-forward) imports, e.g. `from ...rotary import apply_rotary_emb` in the CUDA branch — not traceable by Dynamo, breaks the graph on every per-forward call. Move imports to module scope.
- Passing vLLM/PyTorch Parameter subclasses (ModelWeightParameter, BlockQuantScaleParameter, BasevLLMParameter with custom `__torch_function__`) or NamedTuple-with-ClassVar types (GroupShape) as inputs to custom ops — Dynamo cannot trace through these subclasses. Convert to plain tensors/ints before entering the compiled region.
- Side-effecting tensor-metadata mutations in the traced path (`tensor.requires_grad = False`) or `@compiler.disable`-decorated helpers on the hot path — each forces an unconditional graph break. Remove redundant mutations; gate disables behind `compiler.enable(min_version=...)`.
- Leaving vision/VL encoder submodules or MoE/attention paths in eager mode (missing `@support_torch_compile`) or commenting it out as a temp fix because an inner op (dynamic slicing `q[:, start:end]` driven by cu_seqlens) is untraceable — the whole module falls back to eager.

**显现条件 top3**：
- torch.compile / TorchDynamo (or piecewise CUDA-graph / fullgraph) path is enabled — the break is invisible under eager execution.
- Quantized weight paths (fp8 block-scale / W8A8, NVFP4, marlin int4/int8, MoE) or vision-encoder / rotary / attention paths that dispatch into native or Triton kernels.
- Platform/version-specific branches: ROCm/AITER (MI300, gfx9), Blackwell device-capability checks, or torch version thresholds (torch<2.8 __torch_function__, <2.2 jit path, need >=2.7 for compiler.enable).

**检测时该查什么（父快照核验）**：
- Is this on a per-forward hot path (rotary, attention, linear/GEMM, all-reduce, LoRA shrink/expand) and reached under torch.compile? A break here recompiles/falls-back every step.
- Does the code call into a third-party/native/Triton kernel directly instead of via `direct_register_custom_op`/`torch.library.custom_op` with a `register_fake` meta? Check for missing meta/functional-schema on custom CUDA ops.
- Does it read concrete tensor shapes to Python ints, call `.item()`/`hasattr`/device-capability oracles, or do data-dependent branching inside the traced region?
- Are function-local imports present inside forward branches?
- Are Parameter subclasses / custom `__torch_function__` / NamedTuple-with-ClassVar types passed into the compiled region instead of plain tensors?
- Any side-effecting metadata mutation (`requires_grad=`) or `@compiler.disable` on a hot helper? Check torch version guards (compiler.enable min_version).
- Is the module actually decorated with `@support_torch_compile` (and not commented out as a temp workaround)?

**高发文件 top5**：`vllm/model_executor/layers/quantization/**  (fp8 block-scale, w8a8, nvfp4, marlin MoE apply/process_weights paths)`, `vllm/model_executor/layers/rotary_embedding* and vllm/model_executor/models/*_vl*/vision (ViT attn/rotary wrappers, e.g. vit_attn_wrappers.py)`, `vllm/attention/layer.py and backend files (unified_attention / flash_attn / flashinfer custom-op registration, KV-cache write)`, `vllm/model_executor/layers/fused_moe/** and lora/punica* (MoE + LoRA Triton kernel custom-op wrapping)`, `deepspeed/runtime/zero/** (ZeRO-3 partitioned param coordinator, quantized allgather init)`

### warmup-shape-gap  （39 张 train 卡）

**典型反模式**：
- Warmup/precompile routine uses shapes derived incorrectly under parallelism — e.g. dividing the sequence dimension by tensor-parallel size when the fused kernel actually runs at full seq_length, so the warmed shape never matches the training-hot-path shape and JIT recompiles on the first real step.
- Warmup covers only a subset of runtime states/shapes: e.g. only requires_grad=False (forward) but not the activation-recompute (requires_grad=True) pass, or only single-token decode (query len 1) when speculative decoding submits 1+num_speculative_tokens per seq, so uncovered shapes fall back to eager or slow paths.
- Dummy/warmup input tensors declared with a hardcoded dtype (e.g. torch.bfloat16) or hardcoded context length (max_model_len) that differs from the real runtime dtype/shape, triggering XLA/dynamo/Inductor recompiles or wasted per-capture work.
- Warmup shape space defined via a capture-size or bucket list that is clamped/computed wrong (min(..., max_model_len) using encoder len; max_num_seqs*2 assuming decode=1; buckets exceeding max_num_batched_tokens), so runtime batch/seq sizes exceed the largest captured/precompiled shape and hit uncaptured slow paths.
- Padding runtime shapes to closest precompiled bucket is missing or inconsistent, so dynamically-shaped tensors (logits_indices, num_reqs, sampling tensors, mm encoder inputs) present a fresh shape each step and force recompilation in the hot path.

**显现条件 top3**：
- Parallelism changes the true runtime shape (tensor/sequence/context parallel, expert parallelism) while the warmup shape computation was written for the non-parallel or wrongly-divided case — mismatch surfaces the moment TP/SP/CP > 1.
- A new runtime shape/dtype not covered by warmup appears during serving (new batch size, new num_reqs, speculative-decode query length, multimodal encoder token count, w8a8 fp32 output, block-quant fp8 GEMM M size), triggering JIT/XLA/nvFuser/Inductor recompilation on the first hot-path hit.
- Config default for the capture/warmup shape range is (re)set to a value that no longer spans real workload sizes (cuda_graph_sizes, max_cudagraph_capture_size, dynamic-batching cuda-graph max-tokens), so large-batch or long-context steps miss the captured graph.

**检测时该查什么（父快照核验）**：
- Is the warmup/precompile shape derived from the SAME formula as the hot-path shape? Check for TP/SP/CP division (e.g. seq_length/tp) or hardcoded dtype/context-length that can diverge from runtime.
- Does warmup enumerate ALL states the hot path uses — both requires_grad settings, prefill+decode, speculative query length, every quant/dtype path, every parallelism buffer shape?
- Are runtime dynamic shapes padded/bucketed to a precompiled/captured size, and does the largest bucket/capture size actually cover the configured max (max_num_seqs, max_model_len, max_num_batched_tokens)? Watch clamps like min(x, max_model_len) and encoder-vs-decoder length mismatch.
- Is this warmup on the startup path (default-enabled) or gated behind a flag/backend (nvFuser, DeepGEMM VLLM_USE_DEEP_GEMM, FlashInfer, XLA/TPU, ROCm)? If gated off or moved after capture_model, the pre-JIT benefit is lost.
- Does a config-default change silently shrink the captured/warmup shape range vs the previous default? Compare the pre-change default list/bound against expected workload batch/token sizes.

**高发文件 top5**：`Megatron-LM: megatron/**/initialize.py & pretrain entry (_warmup_jit_function / warmup_jit_function, fused bias_gelu / bias_dropout_add kernels)`, `vllm/v1/worker/gpu_model_runner.py & tpu_model_runner.py (_dummy_run / capture_model / dummy seq_lens & logits_indices warmup)`, `vllm/model_executor/**/deep_gemm_warmup.py & fp8/quant linear methods (_extract_data_from_linear_base_module, warmup_deepgemm_*)`, `vllm/v1/worker/gpu_worker.py / tpu_worker.py (compile_or_warm_up_model, kernel_warmup ordering, _dummy_sampler_run)`, `vllm/config/**/scheduler_config.py & arg-defaults (cuda_graph_sizes, max_cudagraph_capture_size, max_seq_len_to_capture, cuda-graph max-tokens defaults)`

### wasted-compilation-work  （59 张 train 卡）

**典型反模式**：
- Eager/import-time compilation: running a C++/nvcc `load_inline(...)` or JIT build inside an unconditional `if _has_feature:` block at module-import time, so every import pays the compile cost even when the feature is never used on the common path.
- Defining a `@torch.compile`-decorated closure (or Numba/Triton-jitted function) inside a per-call method body, so a fresh compiled function object is created each call and the compile cache never hits.
- Marking context-varying scalar args (token counts, sequence lengths, strides, slice sizes, N/T) as `tl.constexpr` or leaving them specializable in Triton/`@torch.compile`, forcing recompilation for every distinct runtime value; fix is `do_not_specialize=[...]` or dropping constexpr.
- Hardcoded oversized capture/warmup ranges: default CUDA-graph capture sizes like `[512]` / `range(8,513,8)` or DeepGEMM warmup looping every M from a large fixed `CHUNK_SIZE` down to 1, JIT-compiling many kernels for batch/M sizes that never occur given `max_num_seqs`.
- Unconditional expensive compile-time passes: `max_autotune=True`, `coordinate_descent_tuning=True`, C++-fied symbolic shape guards, or `depyf.decompile()` debug dumps enabled for every compiled code object regardless of need.
- Redundant fake-tensor / symbolic-shape recreation (`fake_mode.from_tensor` per input) or `torch.narrow`-style ops that trigger dynamic-shape re-specialization on the compile path.
- Capturing duplicate CUDA graphs per shape (e.g. looping microbatch True/False, or DBO both paths), doubling graph memory and capture time.

**显现条件 top3**：
- torch.compile / Dynamo / Inductor path is active (piecewise or full compile), often gated by torch version (>=2.7/2.8/2.10) and compilation-mode flags; not on eager/no-compilation paths.
- Startup/warmup with CUDA-graph capture or JIT GEMM warmup (DeepGEMM fp8, MoE grouped GEMM) where capture/warmup ranges scale with max_num_seqs, chunk size, or DP world size — worst on tight-GPU-memory or small-batch configs.
- Serving with continuously varying runtime shapes (token counts, num sequences, strides) that hit specialized Triton/compiled kernels, triggering repeated recompilation across batches.

**历史量级样本**：
- DeepGEMM warmup: JIT-compiling one GEMM per M value 'increases the engine startup time by a couple of minutes' (per code comment)
- Default cuda_graph_sizes=[512] captures ~65 graphs up to batch 512 regardless of max_num_seqs
- DBO uniform-decode capture looped microbatch True/False, roughly doubling captured full CUDA graphs per shape

**检测时该查什么（父快照核验）**：
- Is a compile/build (nvcc load_inline, torch.compile, Triton/Numba jit) triggered at import time or per-call rather than lazily on first actual use? Check whether the enclosing block/decorator runs on the default/common path even when the feature is disabled.
- Are compiled/jitted function objects created inside a method body or loop (new identity each call) instead of at module scope, defeating the compile cache?
- For Triton/torch.compile kernels: are context-varying scalars marked constexpr or left specializable? Are they in a hot serving path with variable shapes?
- For warmup/capture loops: is the size range derived from actual limits (max_num_seqs, real chunk size) or a hardcoded large default? Count how many kernels/graphs get built for sizes that can never occur.
- Are expensive compile-time-only passes (max_autotune, coordinate_descent_tuning, cpp_symbolic_shape_guards, depyf decompile) enabled unconditionally, and is their runtime benefit actually realized (e.g. guards skipped anyway)?
- On AOT/cache-hit paths: is any redundant tracing / fake-tensor regeneration / dynamic-shape re-specialization still executed?
- Is the same graph/kernel captured multiple times per shape (microbatch variants, DBO both paths, redundant options loops)?

**高发文件 top5**：`vllm/compilation/** (support_torch_compile decorator, backend wrapper, inductor config, bytecode hooks)`, `vllm/config/** (CompilationConfig, SchedulerConfig — cuda_graph_sizes / cudagraph_capture_sizes)`, `vllm/model_executor/**/fused_moe/** and DeepGEMM warmup / Triton kernel modules`, `vllm/v1/**/gpu_model_runner / cudagraph manager (capture-size, decode-path graph capture)`, `Megatron-LM inference unified_memory.py / dynamic_context.py (import-time load_inline UVM allocator)`


## 大类：concurrency-sync

### blocking-d2h-sync  （98 张 train 卡）

**典型反模式**：
- Calling `.item()` / `.tolist()` / `.cpu()` on a GPU tensor inside a per-step or per-microbatch hot path (metadata build, loss/grad-norm reduction, timer/logging), forcing a device→host sync every iteration.
- Constructing a tensor from a Python list whose elements are GPU scalars, e.g. `torch.tensor([gpu_scalar_a, gpu_scalar_b], device=...)`, or `torch.tensor(1.0)` on CPU then multiplying with a GPU tensor — each element read forces a D2H (or implicit H2D) sync.
- Using a GPU tensor scalar as a Python slice bound / loop range / index inside a `for` loop (e.g. `x[:, cu_seqlens[i-1]:cu_seqlens[i]]`, `range(len_gpu_tensor)`), forcing one sync per iteration.
- Copying a CPU-side length/index tensor to device with a plain blocking `.to(device)` (non-`non_blocking`, no pinned memory) on every forward, or eagerly materializing a `*_cpu` shadow field (seq_lens_cpu, num_computed_tokens_cpu) that is not actually needed on the decode fast path.
- Per-parameter Python loop computing norms (`torch.norm(g).item()`) or host-side histograms (`np.histogram(sorted.cpu(), ...)`) instead of a single fused GPU reduction.
- Evaluating a CUDA boolean tensor inside a Python `assert`/`if` (e.g. `assert not loss.isnan()`) every iteration, which implicitly syncs.

**显现条件 top3**：
- Per-step / per-microbatch hot paths: attention metadata build (decode/spec-decode), MoE token dispatch/permutation, loss & grad-norm reduction, and forward_step scale construction.
- Async or overlapped scheduling regimes where the sync serializes the launch queue and defeats CPU/GPU overlap (vLLM async_scheduling, spec-decode, CUDA graphs, pipelined training).
- Parallelism that scales the sync cost: MoE with EP/TP, context/decode-context parallelism (CP/DCP>1), data parallel (dp_size>1), pipeline parallelism, multi-rank distributed logging.

**检测时该查什么（父快照核验）**：
- Is this code on a per-step / per-microbatch / per-request hot path (metadata build, dispatcher preprocess, forward_step, loss/grad reduction, logging), or a rare setup path?
- Does it call `.item()`, `.tolist()`, `.cpu()`, `.numpy()`, `np.histogram(x.cpu())`, or read a GPU scalar as a Python value (slice bound, loop range, list element in torch.tensor([...]))?
- Are `*_cpu` shadow tensors (seq_lens_cpu, num_computed_tokens_cpu) eagerly materialized when the fast path doesn't need them? Is there a guard (needs_seq_lens_cpu) or lazy invalidation instead?
- Is a CPU->device copy blocking (plain `.to()` without non_blocking/pinned memory) instead of overlapped?
- Is a CUDA bool tensor evaluated inside a Python assert/if branch each iteration (NaN checks), and is it default-enabled?
- Under EP/TP/CP/DCP/DP/PP, does the sync scale with active-request/rank/expert count, and can it be deferred (cuda_sync_point state machine) or moved off-stream?

**高发文件 top5**：`vllm/v1/attention/backends/** (metadata builders: FlashAttn/FlashInfer/MLA/GDN, CommonAttentionMetadata seq_lens_cpu)`, `megatron/core/transformer/moe/** (token dispatcher preprocess, GroupedMLP/SwitchMLP permutation)`, `megatron/core/**/forward_step / pipeline schedules (aux-loss scale, grad_scale_func)`, `megatron/optimizer/clip_grads.py & pretrain_*/utils.py (grad-norm loop, NaN asserts, context-parallel loss/index tensors)`, `vllm/v1/spec_decode/** (eagle.py, medusa proposer; spec-decode draft seq_lens / argmax)`

### defensive-oversync  （19 张 train 卡）

**典型反模式**：
- Placing a Python `assert`/`if` on a device (GPU/TPU) tensor value inside a hot forward/step path — evaluating the boolean forces an implicit device→host synchronization every iteration.
- Calling a full device-wide sync (`torch.cuda.synchronize()`, `get_accelerator().synchronize()`, `xm.mark_step()`/`wait_device_ops()`, `cudaStreamSynchronize`) unconditionally after an async op (p2p exchange, all-gather, reduction dispatch, profile run) instead of relying on stream/event ordering.
- Leaving diagnostic/timing scaffolding (barriers, per-step throughput/wall-clock timers that sync, `torch.distributed.barrier()`) enabled by default in the training/inference loop rather than gating it behind an opt-in flag.
- Emitting redundant intra-warp barriers (`__syncwarp()`, extra `cg` tile syncs) around reductions that already imply full-warp lane synchronization via `__shfl_*_sync` with a full mask.
- Collective calls (`dist.barrier()`) inside per-rank loops or conditional paths where not every rank participates in lockstep, causing straggler stalls or hangs.

**显现条件 top3**：
- Multi-rank distributed training with collectives (pipeline/tensor/data parallel, ZeRO): one rank's sync/GC pause or non-participating barrier stalls all peers; cost scales with rank count.
- Hot per-step path (forward, decode step, p2p exchange, optimizer step) where the sync/barrier/assert executes every iteration, serializing CPU↔device and creating bubbles.
- Small-message / small-batch or short-kernel regimes (small GEMM, warp reduction, tens-of-KB allreduce) where fixed sync overhead dominates useful work.

**历史量级样本**：
- CPU SHM allreduce (<1MB): 'allreduce latency will reduce 30% to 50%'; e.g. 32768 bytes: 16.13us new vs 25.17us old

**检测时该查什么（父快照核验）**：
- Is this on a hot path (per-step / per-p2p / per-forward)? A sync that is fine once at startup is a regression when hit every iteration.
- Does the guard read a *device* tensor value (assert/if on GPU/TPU data)? That forces an implicit device→host sync — move it off the hot path or make it opt-in/debug-only.
- Is the sync/barrier unconditional and default-enabled? Check whether it can be gated behind a flag or replaced by stream/event ordering (record/wait) or a properly-masked warp primitive.
- Is it a full device-wide `synchronize()` where a per-stream/per-handle event wait would suffice? Full sync serializes ALL device work including unrelated compute.
- For collectives (barrier/all-gather in loops or conditionals): do ALL ranks reach it in lockstep? Non-uniform participation causes straggler stalls or hangs.
- Is the barrier merely for timing/diagnostics? If so it should not ship enabled by default.
- Is a warp/block barrier redundant given full-mask `__shfl_*_sync` or cooperative-group semantics already synchronizing the lanes?

**高发文件 top5**：`megatron/**/p2p_communication.py (pipeline p2p _communicate/communicate)`, `**/schedules.py, **/training/train_step (pipeline schedule + barrier/timing sites)`, `deepspeed/runtime/zero/** (ZeRO stage 1/2/3 optimizer, param init all-gather, average_tensor)`, `deepspeed/runtime/**/timer / engine.py (ThroughputTimer, SynchronizedWallClockTimer, pipe engine)`, `csrc/**/*.cu kernels (MoE grouped top-k, Marlin GEMM, fp8 kv-cache indexer warp reductions)`

### gil-held-blocking-call  （16 张 train 卡）

**典型反模式**：
- Calling a CPU-bound function (tokenization, chat-template rendering, multimodal image/video/audio decode, input mapping of PIL→tensor) directly inside an `async def` coroutine, so the entire asyncio event loop is blocked for the whole duration and all other concurrent requests stall.
- In pybind11/C++ extension entry points, invoking blocking native code or async CUDA kernel-launch functions while still holding the Python GIL (no `py::call_guard<py::gil_scoped_release>`), serializing multi-threaded Python callers and, on cleanup paths that re-acquire the GIL, deadlocking.
- Running the HTTP/API frontend and the engine/execution loop in the same Python process (or the same thread), so they contend on the GIL and degrade TTFT/inter-token latency and throughput under load.
- Using a single-thread/single-worker ThreadPoolExecutor or a Python for-loop to serialize per-item work (per-file writes, per-sequence logits processors, dequeue of worker responses), where the GIL prevents any real overlap.
- Awaiting a synchronous blocking worker/executor call inline (`output = executor(*args)`) on the async path instead of offloading to a thread/process, starving other coroutines.

**显现条件 top3**：
- Async server (OpenAI-compatible / FastAPI) under concurrent request load, where one request's CPU-bound preprocessing blocks the shared event loop and inflates TTFT/inter-token latency for all in-flight requests.
- Large or heavy inputs amplify the block: very large prompts (long tokenization), multimodal image/video/audio inputs (10-50 ms input-mapping / decode spikes), scaling with request concurrency.
- Multi-threaded / multi-process execution where the GIL serializes work that should overlap: C++/CUDA kernel launches from extension threads, comm/overlap threads, dataloaders, distributed-checkpoint parallel file writes, or driver dequeue of worker-response queues.

**历史量级样本**：
- 10-50 ms latency spike per request from synchronous multimodal input mapping (PIL→tensor) inside the engine-core loop (from removed FIXME in core.py)
- ~0.5 ms spent per decode blocking the network thread; issue-measured schedule_time_us: 118, execute_time: 7664, postprocess_time_us: 409
- Test exercises ~300k-token inputs where long tokenization blocks the async event loop

**检测时该查什么（父快照核验）**：
- Is a CPU-bound or blocking call (tokenizer, chat template, media decode, input mapper, HF encode) invoked directly inside an `async def` without `run_in_executor`/thread/process offload?
- Does the code path run on a hot, shared event loop or engine-core loop that also drives scheduling/execution for all requests?
- For native/pybind11 entry points: is the function registered without `py::call_guard<gil_scoped_release>` while it makes blocking or async CUDA/kernel-launch calls? (check whether cleanup/refcount paths need the GIL → deadlock risk)
- Do frontend and engine loops share one process/thread and thus contend on the GIL under load?
- Is 'parallelism' actually achieved, or is a single-thread ThreadPoolExecutor / Python for-loop serializing work the GIL already blocks?
- Is the blocking behavior default-enabled (on by default) vs opt-in via an env var / config flag (e.g. thread-count, pool-size, async offload)?

**高发文件 top5**：`vllm/entrypoints/openai/** (serving/chat/completion async request handlers, tokenization/preprocess)`, `vllm/v1/engine/core.py and vllm/v1/executor/** (engine-core loop, MultiprocExecutor response queues)`, `vllm/multimodal/** and media_io loaders (async fetch/decode of image/video/audio)`, `**/pybind.cpp and PyTorch C++ extension entry points (pybind11 registrations, call guards)`, `Megatron-LM distributed-checkpoint save strategy (Torch-Dist FileSystemWriter / async writer)`

### host-scalar-loop-work  （57 张 train 卡）

**典型反模式**：
- Per-request/per-element Python loop over batch entries that touches GPU scalars: `.tolist()` then a dict-build/O(n) grouping, or indexing a CUDA tensor for one scalar per iteration — each access an implicit device→host sync in a per-step hot loop.
- Materializing small tensors from Python lists on host and implicitly copying to device: `torch.tensor([0], device='cuda')` / `torch.FloatTensor([0]).to(gpu_tensor)` instead of allocating directly on device (`torch.zeros(1, device='cuda')` / `torch.cuda.FloatTensor([0])`).
- Serial CPU preprocessing built from many tiny torch ops (`torch.arange`/`repeat_interleave`/`cumsum`/`split`/`cat`/`stack`/`.view`/`.expand`) per request, each with per-op overhead and kernel launches, where a single numpy/vectorized/fused-kernel computation would suffice.
- Per-parameter or per-sequence reduction loop that forces sync each step (`param_norm.item()` accumulate, `for i in range(size): ret[i, start:end] = True`, `hidden_states.split(prompt_lens.tolist())` + `forward_one` per seq) instead of one batched/vectorized/multi_tensor op.
- Scalar element-by-element compute loops in CPU SIMD backends (`for lane: std::exp/std::tanh/std::erf`) or host-side element-wise encoders left un-vectorized (TODO-marked).

**显现条件 top3**：
- Decode/generation hot path with per-step host work whose cost scales with number of active requests / batch size (dynamic in-flight batching, spec-decode, pooling, per-request sampling metadata).
- CUDA workloads where a Python-side scalar loop or list→tensor allocation triggers implicit host↔device copies/syncs each call or microbatch (gradient clipping, pipeline schedule tensors, EPLB rebalancing).
- Multi-sequence / packed-sequence / multimodal preprocessing (THD RoPE, M-RoPE position ids, vision cu_seqlens, chunk masks) where per-sequence loops or many tiny tensor ops dominate over the actual compute.

**历史量级样本**：
- Up to 8x lower overhead (ngram spec-decode drafter, per-request Python loop replaced by single batched call)

**检测时该查什么（父快照核验）**：
- Is this on a per-decode-step / per-microbatch / per-request hot path, or is it one-time setup? Scalar host loops in the steady-state path are the dangerous ones.
- Does the loop body index a CUDA tensor for a scalar, call `.item()`/`.tolist()`/`.cpu()`, or build a tensor from a Python list — i.e. force an implicit device→host sync per iteration?
- Does the cost scale with batch size / num active requests / sequence length / num params (O(n) Python iterations)? Prefer a single batched/vectorized/fused-kernel or numpy path.
- Is a small helper tensor allocated on host then copied to device (`torch.tensor([...], device='cuda')`, `.to(gpu_tensor)`)? Allocate directly on device.
- Is the caller default-enabled (e.g. metrics/accuracy computed unconditionally every step, logprobs materialization)? Can it be gated or deferred?
- Could the per-element loop be handed to a vectorized op / numba-batched / multi-threaded (cpu_count//2) / SIMD path instead of a serial Python or scalar C++ loop?

**高发文件 top5**：`vllm/v1/worker/**, vllm/v1/**/input_batch.py (per-request host bookkeeping, input prep)`, `vllm/v1/spec_decode/**, vllm/v1/sample/**, logprobs/pooling processors (per-request loops, tolists)`, `megatron/**/optimizer/** & clip_grads / grad-norm helpers (host list→cuda tensor, per-param loops)`, `megatron/core/**/schedules.py (pipeline free_output_tensor scalar alloc); dynamic inference engine sampling map`, `vllm model-specific encoders: qwen*-vl vision, deepseek_v3.2 sparse indexer, mrope/rope position helpers, MoE/EPLB rebalance`

### shared-mutable-state-contention  （10 张 train 卡）

**典型反模式**：
- Storing a per-call/per-GEMM selector or config on a shared mutable object (e.g. a field on self.config reused across submodules/layers), so a later write clobbers what an earlier consumer expected — the config becomes a de-facto shared latch instead of a value passed explicitly per site.
- Lock-free single-producer/multi-consumer shared-memory queues that assume 'no synchronization needed' — writing a ready/flag byte and payload without a memory barrier, so consumers on weak memory-order CPUs observe the flag before the data (torn/stale reads, hang).
- Cross-CUDA-stream sharing of a tensor without record_stream/event synchronization (e.g. cloning or producing a tensor inside a side stream and consuming it on the default stream), letting the allocator reuse memory while the other stream still reads it.
- Concurrent entry into a critical section that issues collectives (torch.distributed.broadcast / generate) — e.g. Flask app.run(threaded=True) letting multiple requests interleave collectives across ranks, causing deadlock.
- Multiple accumulators atomic_add-ing into a single shared output buffer (SPLIT_K into one (M,N) tensor) instead of per-partition buffers, causing atomic memory contention.
- Multiple coordinators/registries keyed on some mode (training True/False) each keeping a private in-flight registry while sharing one global status field, so concurrent modes see inconsistent shared state.

**显现条件 top3**：
- Tensor/pipeline/expert parallelism with shared communication or memory infrastructure (TP comm-overlap userbuffers, shm ring buffer over multiprocessing.shared_memory, DP/EP under load).
- A shared object/field/buffer is written by one path and read by another that assumed a stable value — especially when submodules, layers, streams, or training/eval modes concurrently touch it.
- Weak memory-order or GPU-specific conditions (aarch64/ARM CPUs for missing barriers; CUDA side-streams for missing record_stream; AMD GPUs for atomic contention; SM100 warp barriers using arrive-without-wait).

**历史量级样本**：
- AWQ Triton SPLIT_K per-split buffer fix: 2-5x speedup on AMD MI300 (minor on MI250)

**检测时该查什么（父快照核验）**：
- Is a config field / module global / buffer being mutated in-place and read by multiple consumers (layers, submodules, streams, ranks)? Prefer passing the value explicitly per call site instead of latching on shared state.
- For shared-memory / lock-free queues: is there a memory barrier between the payload store and the readiness-flag store (and matching load fence on the reader)? Assume weak memory order (ARM).
- For CUDA multi-stream overlap: does every tensor produced on one stream and consumed on another have record_stream() or an event to prevent premature allocator reuse?
- For barriers/collectives: does every arrive() have a matching wait() (arrive_and_wait), and is the critical section that issues collectives serialized (no threaded=True concurrent entry)?
- For accumulation kernels (SPLIT_K etc.): is there atomic_add into one shared output? Consider per-partition buffers to avoid contention (hot on AMD).
- Are separate coordinators/registries keyed on a mode while sharing a single global status? Check for cross-mode interference when both modes are active.

**高发文件 top5**：`Megatron-LM: text-generation inference server (Flask app.run) and TE comm-overlap config plumbing (transformer/*, config dataclasses)`, `vllm/distributed/device_communicators/shm_broadcast.py (ShmRingBuffer/MessageQueue)`, `vllm MoE layers with shared-experts stream overlap (model_executor/layers/fused_moe/*)`, `vllm quantization Triton GEMM kernels (AWQ split-k; model_executor/layers/quantization/*) and CUTLASS MLA attention kernels (SM100)`, `DeepSpeed ZeRO-3 partitioned parameter coordinator / partition_parameters (runtime/zero/*)`

### sm-margin-reservation  （8 张 train 卡）

**典型反模式**：
- Kernel launch hardcodes its SM count to the full device count (e.g. `num_sm = GetMultiProcessorCount()` or passing literal `0` as the margin arg) so no SMs are reserved for concurrent comms, forcing the norm/GEMM kernel to contend with overlapped communication.
- SM-margin env var (e.g. FWD/BWD margin knob) is read/parsed on the Python side but the value is never threaded into the C++ kernel-launch path, so the requested reservation silently has no effect.
- Inference or fp8/fp8-variant kernel paths get forgotten during a margin rollout: only the training/non-fp8 path subtracts the margin while the inf/fp8 twin keeps the hardcoded full-SM count, causing asymmetric contention.
- Introducing an alternative comm backend or dispatcher (UCC/HybridEP) or new PG options (CGA cluster/CTA bounds) as opt-in without wiring the SM-reservation/connection knobs, so overlap either doesn't materialize or oversubscribes SMs.

**显现条件 top3**：
- Tensor/sequence-parallel comm-compute overlap where a non-zero SM margin is set but not honored.
- fp8 and non-fp8 norm/GEMM paths incl. inference variants on NVIDIA GPUs deriving SM count from multiProcessorCount.
- MoE EP or pipeline-parallel with alternative comm backends/dispatchers needing overlap-enabling connection settings.

**检测时该查什么（父快照核验）**：
- At the kernel-launch site, confirm the SM count passed is `multiProcessorCount - sm_margin` and not a hardcoded literal (0) or the raw full device count.
- Verify any Python-side SM-margin / margin env var is actually threaded through the descriptor into the C++ impl and consumed at launch (not read-then-dropped).
- Check parity across all variants: fwd/bwd, fp8/non-fp8, and inference paths must all apply the same margin logic.
- Confirm the feature is default-enabled vs opt-in: env-var/config knobs (margin, NCCL PG options, comm backend) may leave overlap unrealized if defaults reserve zero SMs.
- For overlap symptoms, check whether it is a hot path (per-layer norm/GEMM during every fwd/bwd) and whether comm-overlap is actually the caller.

**高发文件 top5**：`TransformerEngine: transformer_engine/common/ (layernorm / rmsnorm impl, e.g. LayerNormForwardImpl/BackwardImpl)`, `TransformerEngine: C++ pytorch extensions for layernorm/rmsnorm fwd/bwd incl. fp8 and *_inf variants`, `TransformerEngine: te_gemm / comm-gemm-overlap userbuffers path (NVTE_EXT_MARGIN_SM)`, `Megatron-LM: megatron/core/ parallel-state / process-group setup (get_nccl_options, new_group pg_options)`, `Megatron-LM: MoE flex token dispatcher fused_a2a.py (HybridEP dispatch/combine, init buffer)`

### spin-wait-hang  （18 张 train 卡）

**典型反模式**：
- Busy-wait poll loops that spin (`while cond: time.sleep(tiny)` / `sched_yield()` / `await asyncio.sleep(0)`) with no idle backoff, burning a full CPU core per worker even when there is no work; also re-invoking a step() that does no real work when the only pending requests are blocked on an external event.
- Reading/checking the shared condition (buffer size, ready flag, state counter) OUTSIDE the lock/memory-scope that the producer mutates it under, so the waiter observes a stale value and never wakes — or the peer races ahead into the next round (resetting a state flag) before the waiter observes the terminal value.
- Cross-rank / cross-pipeline-stage collectives (broadcast / p2p send/recv) issued inside a per-step loop with no shared synchronization point, so ranks/stages desynchronize on divergent iteration counts and one side blocks forever on a mismatched collective.
- Blocking on a socket/queue with an unbounded or coarse timeout for control flow (`queue.get(timeout=...)`, `socket.recv_multipart()`, `asyncio.wait_for(...)`) and relying on context teardown / timeout-exceptions to unblock shutdown — racy, and stalls the hot loop for the full timeout each idle iteration.
- Wait/handshake protocols that use device-scoped atomics or a slow-joiner-prone pub/sub READY signal, so the waiter never sees the write that should release it (dropped subscription message or non-coherent flag).

**显现条件 top3**：
- Multi-process / multi-rank distributed execution (pipeline-parallel, tensor-parallel, disaggregated prefill/decode) where two or more participants must rendezvous via collectives, p2p, shared memory, or ZMQ — hang requires the participants' iteration counts or flag states to diverge.
- Idle or between-request periods on the async/serving hot path, where a spin loop pins a CPU core at 100% or an empty-queue blocking get stalls the engine loop for the full polling timeout.
- External-event-gated states (remote KV handshake, buffer-full backpressure, socket/shutdown teardown, pathological regex with no timeout) that leave a thread parked or spinning indefinitely without progress.

**检测时该查什么（父快照核验）**：
- Is the wait a spin/busy-poll (fixed tiny sleep, sched_yield, sleep(0)) rather than a blocking wait with backoff or a condition-variable/event wake? If so, does it pin a CPU thread when idle?
- Is the loop condition (buffer size, ready/done flag, state counter) read under the same lock / memory scope the mutator uses? Are device-scope atomics used where system-scope coherence is required?
- For cross-rank collectives/p2p inside a per-step loop: is there a shared synchronization point guaranteeing all ranks/stages execute the same number of collectives, even on early-exit / multiple-invocation paths?
- Does the step()/engine loop re-run when model_executed is False but unfinished requests are only blocked on an external handshake — i.e. does it back off instead of hot-spinning?
- On shutdown/error, is the blocking recv/get guaranteed to be unblocked by explicit socket close (not just ctx.destroy/linger) or a timeout? Is asyncio.wait_for used on a Python version subject to the cancellation-race bug?
- Are external-input-driven operations (regex match, buffer waits) bounded by a timeout, or can adversarial/slow input hang the handling thread indefinitely?

**高发文件 top5**：`vllm/**/shm_broadcast.py (MessageQueue shared-memory ring buffer, spin/handshake paths)`, `vllm/**/engine/*core*.py, async_llm_engine.py (EngineCore step / run_engine_loop / MPClient)`, `vllm/**/kv_transfer/**/*buffer*.py (disaggregated KV SimpleBuffer producer/consumer)`, `TransformerEngine userbuffers comm-overlap sources (userbuffers.cu / initialize_ub, push/recv sync flags)`, `DeepSpeed CPU SHM all-reduce + PipelineEngine (inference_all_reduce state-wait, eval_batch p2p schedule)`

### stream-serialization  （27 张 train 卡）

**典型反模式**：
- Launching a CUDA kernel with the default triple-chevron `<<<blocks, threads>>>` (or a stream pulled from `getStreamFromPool`) instead of `getCurrentCUDAStream()`, so the kernel lands on stream 0 / an unrelated pool stream with no ordering relationship to surrounding work — cannot overlap and may run without proper dependency ordering.
- Synchronous host->device copies in per-step hot paths: `tensor.to(device)` / `torch.tensor(list, device=device)` / advanced-indexing a GPU tensor with a Python list, which forces a blocking H2D transfer and cudaStreamSynchronize, stalling CPU kernel pre-launch and creating a decode/step bubble.
- Serializing communication and compute on the same stream/process group: issuing both P2P directions on one NCCL group at PP==2, running RCCL/NCCL collectives on the default stream, or omitting the high-priority-stream option so comm kernels can't preempt overlapping compute.
- Running independent workstreams sequentially on the main stream (shared-experts vs routed MoE experts, gate/router before experts, single offload stream per direction, single H2D param-copy stream in offload) instead of a dedicated `torch.cuda.Stream()` with proper wait_stream/record ordering.
- Polling a CUDA-resident flag (`cuda_tensor == 1`) every iteration, forcing device->host sync each step.

**显现条件 top3**：
- Comm+compute overlap paths (TP async allreduce/sequence-parallel, PP P2P, Userbuffers GEMM+comm, FSDP comm groups) where stream priority, CUDA_DEVICE_MAX_CONNECTIONS, or process-group ordering is misconfigured — comm and compute serialize onto the same HW queue.
- MoE / offload workloads where independent branches (shared experts, routed experts, H2D/D2H tiles, grouped quantize) could run concurrently but are issued on one stream — overlap disabled under EP or flashinfer-cutlass+DP>1 guards.
- Per-step input-prep / metadata-build / sampling / guided-decoding hot paths on CUDA GPUs (with pinned memory available) where blocking H2D copies stall CPU kernel launch and open a GPU bubble each step.

**检测时该查什么（父快照核验）**：
- Is this on a per-iteration / per-step hot path (attention metadata build, sampler, input prep, readiness polling)? Blocking transfers or syncs here cost a bubble every step.
- For every `.to(device)` / `torch.tensor(..., device=)` / GPU advanced-index-by-Python-list in the hot path: is `non_blocking=True` used AND is source pinned host memory (is_pin_memory_available; excluded on WSL/XPU/Neuron/CPU)?
- For every raw kernel launch (triple-chevron in .cu, or torch stream selection): is it on `getCurrentCUDAStream()` / current_stream()? Check for `<<<...>>>` without a stream arg or `getStreamFromPool`.
- Does the caller rely on comm/compute overlap? Verify CUDA_DEVICE_MAX_CONNECTIONS is set correctly for the parallelism (=1 for TP async path; ordering events when >1 on Hopper) and that is_high_priority_stream / stream priorities are set on comm process groups.
- When two logically independent branches exist (shared experts vs router, both P2P directions, offload directions, grouped quantize), are they on distinct streams with correct wait_stream/record_stream ordering, or serialized on the main stream?
- Are overlap optimizations silently disabled by platform/config guards (is_cuda vs is_cuda_alike for ROCm; EP-enabled; flashinfer-cutlass+DP>1; default stream 0 on ROCm)?
- Any per-iteration read of a CUDA-resident scalar/flag that forces device->host synchronization?

**高发文件 top5**：`vllm: model_executor/layers/fused_moe/** (shared_experts stream, SharedFusedMoE, forward_impl)`, `vllm: v1/attention/backends/** and worker/model_runner input-prep / metadata builders (flash_attn.py, flashinfer.py, utils.py)`, `vllm/**/sampler + guided/structured-decoding + distributed (pynccl/current_stream) paths; *.cu quant kernels (awq, scaled_fp4)`, `Megatron-LM: examples/*pretrain*.sh (CUDA_DEVICE_MAX_CONNECTIONS), core/ process-group / pipeline P2P + NCCL options setup`, `TransformerEngine: common/ CUDA quantize kernels (mxfp8/blockwise, PDL/stream serialization), CommOverlapBase / userbuffers, offload handler (AsyncDoubleBufferGroupOffloadHandler)`


## 大类：config-observability

### batch-composition-mismatch  （21 张 train 卡）

**典型反模式**：
- Computing batch admission bounds against the wrong unit: counting seq_groups (len(self.running)) when the limit (max_num_seqs) is meant to bound sequences, so multi-sequence groups (beam search / parallel sampling n>1) silently over- or under-fill the batch.
- Building a per-batch tensor/params structure sized by a static/default dimension (e.g. `[value] * max_batch_size` or per-chunk token budget left at chunked-prefill default) that no longer matches the actual current batch size after in-flight/dynamic batching was introduced.
- Constructing a set/count of composition keys that unintentionally includes a sentinel/no-op member (e.g. lora_int_id==0 for no-LoRA requests) so a mixed batch overcounts distinct members against a small cap.
- Iterating over a mixed inflight batch under an assumption it is homogeneous (all-prefill, or vision-model-present on every rank), so collectives / KV recv paths misfire when the batch actually contains decode requests, text-only samples, or empty/no-work slots.
- Coupling admission gates to unrelated queue state (e.g. blocking brand-new requests while any request is paused), so composition of the running set stalls forward progress under memory pressure.

**显现条件 top3**：
- Mixed prefill+decode (in-flight / continuous / disaggregated) batching where scheduler code assumes a single request type or a static batch size.
- Multi-sequence sampling (beam search, best_of>1, parallel n>1) where one seq_group maps to many sequences and admission/estimation counts the wrong granularity.
- Batch-width / token-budget defaults (max_num_seqs, max_num_batched_tokens, per-chunk budget) that shift or mismatch across engine versions, model types (MLA, multimodal), or hardware tiers.

**历史量级样本**：
- 12K context length under 0.6.x -> cannot exceed 3K on 0.7.0 V1 (max_num_seqs default reverted 1024->256 for non-H100/H200; TP=4, INT4/AWQ/INT8)

**检测时该查什么（父快照核验）**：
- Is this code on the scheduler hot path (per-step admission, batch assembly, or per-iteration tensor sizing)?
- Does the admission/estimation count sequences vs sequence-groups consistently with the limit it is compared against?
- Are per-batch tensors/params sized from actual current_batch_size, or from a static max_batch_size / stale default?
- Does any count/set include sentinel values (0 lora id, dummy [1,1] image, num_tokens=1 dummy) that skew composition?
- Does the batch-loop assume homogeneity (all-prefill, vision-present-on-all-ranks) that breaks under mixed/inflight batching and can cause hang on collective paths?
- Is a default token/seq budget (max_num_batched_tokens, max_num_seqs, per-chunk) branched by model type or hardware tier, and did that default recently change?

**高发文件 top5**：`vllm/core/scheduler.py (admission loops, curr_loras/num_curr_seqs, chunked-prefill ordering)`, `vllm/config.py / SchedulerConfig (max_num_batched_tokens, max_num_seqs defaults, MLA/multimodal floors)`, `vllm/v1/... scheduler & GPUModelRunner.execute_model (chunked mm input, dummy DP forward)`, `vllm distributed/kv_transfer connector recv paths (disaggregated prefill/decode)`, `Megatron-LM inference context (DynamicInferenceContext.check_availability, flash-decode sequence_len_offset / current_batch_size)`

### boolean-guard-misfire  （91 张 train 卡）

**典型反模式**：
- Using a value that is truthy/falsy in a way that inverts the intended gate: e.g. a single-element zero CUDA tensor (`if overflow_buf:`), an integer size stored where a boolean was intended (`self.flag = args.pipeline_model_parallel_size` then `if self.flag:`), so PP==1 or size!=0 always takes the wrong branch.
- Comparing against a 'disabled' sentinel (-1) with `>=` or `>`: `batch*seqlen >= threshold` where threshold defaults to -1 makes the guard always true, silently forcing the microbatched/pipelining path on every call.
- Gating a fast/fused/optimized path on the wrong config flag or a related-but-distinct flag (`nccl_ub` instead of `fsdp_double_buffer`; `supports_chunking()` instead of the env-aware `enable_chunking()`; only-`is not None` on a stream that is valid on unintended platforms), so the optimization is skipped or wrongly enabled.
- Guard that conflates local vs global index / padded vs unpadded count / blocking vs non-blocking mode: passing local layer index where global is required, comparing SP-padded token count for uniform-decode classification, `if not is_done and blocking:` never running cleanup on the non-blocking path.
- Silently changing a default when refactoring a boolean gate: replacing an implicit environment/context check with an explicit parameter defaulting to False/True (e.g. `enable_autocast=False`, `args.add_bias_linear` reused to gate `bias_dropout_fusion`, argparse `default` inherited from a config attribute), dropping a previously-active fast path.
- Removing or relocating a `if self.pre_process:` / `and rotary_pos_cos is not None` style guard, causing unconditional allocation of large modules/tensors or skipping tensor creation needed by a downstream fast path.

**显现条件 top3**：
- Sentinel/default value collides with the comparison operator (threshold=-1 with `>=`, PP size==1 as truthy) so the guard fires (or never fires) regardless of actual runtime shape/batch — worst on the inference/generation microbatch-routing and pipelining paths.
- Feature is enabled on a config the flag was not meant to cover: fp8/bf16 mixed precision, MoE with shared experts, custom FSDP with double-buffer, sequence/pipeline/virtual-pipeline parallelism, or a non-CUDA cuda-alike platform (ROCm) where an `is not None` stream check passes.
- Torch.compile with dynamic/symbolic shapes where evaluating a SymInt equality in boolean context installs a specialization guard, forcing recompiles/compile-time blowup.

**检测时该查什么（父快照核验）**：
- At the parent snapshot, check every `if <flag>:` where `<flag>` may be a tensor, an integer size, or an env-derived helper — confirm the intended boolean semantics and that a 'disabled' sentinel (-1/0/None) is compared with the correct operator.
- Is this a hot path (per-iteration forward/backward, per-microbatch routing, inference decode, gradient all-reduce/reduce-scatter, CUDA-graph capture)? A misfired guard here flips a fast/fused path on/off every step.
- Is the guarded optimization default-enabled? Check whether flipping the branch silently drops FP8 weight caching, fused bias-dropout, CUDA-graph buffer reuse, autocast, or async param all-gather.
- Trace the flag to its source: does it come from a distinct-but-related config attribute (nccl_ub vs fsdp_double_buffer), a local vs global index, a padded vs unpadded count, or an argparse default inherited from a config object?
- Check the caller/platform matrix: does the guard behave correctly for PP==1 and PP>1, TP==1 and TP>1, CUDA vs ROCm, blocking vs non-blocking, fp16 vs bf16, static vs dynamic shapes?

**高发文件 top5**：`megatron/core/inference/**/forward_step*.py (ForwardStep / run_one_forward_step pipelining & threshold guards)`, `megatron/core/transformer/{transformer_layer,transformer_block}.py (fused kernel / fp8 / bias-dropout / index guards)`, `megatron/core/pipeline_parallel/schedules.py (no_sync / async param all-gather / autocast guards)`, `megatron/core/models/**/gpt_model.py (_preprocess, embedding/pre_process guards, CUDA-graph flags)`, `vllm/model_executor/layers/fused_moe/*.py and vllm/v1/worker/**/gpu*.py (chunking/enable helpers, uniform-decode/padding, shared-experts stream guards)`

### config-toggle-perf-feature  （149 张 train 卡）

**典型反模式**：
- Flipping a performance-affecting flag's default from opt-in (store_true) to opt-out (store_false, dest=...) — or vice versa — so that fusion/overlap/parallelism paths silently change for all existing jobs that never set the flag.
- Hardcoding a config knob ON (e.g. wrapping a forward pass in an unconditional fp8/autocast context or force-setting an env var) with no off-switch, forcing every run through the new precision/comm path regardless of user intent.
- Auto-mapping args→config-dataclass fields by name-matching, where a dataclass field defaults to True but has no matching CLI arg, so the field can never be reflected/toggled from the command line and silently stays at its default.
- Adding a new CLI-tunable that sizes a hot resource (concurrency task count, microbatch split threshold, sync interval) with a default value that changes batching/pipelining behavior vs the previously-hardcoded value.
- Adding new config params to a builder/constructor call site but not forwarding them to the underlying context/engine, so it silently falls back to constructor defaults (feature toggled but ineffective).

**显现条件 top3**：
- A feature flag's default changes (opt-in↔opt-out), so jobs that don't explicitly set it inherit a different fusion/overlap/precision/parallel path — most visible on the workload the feature targets (e.g. SwiGLU+RoPE fusion, TP-comm-overlap, fp8/autocast, fully-parallel checkpoint save).
- The toggle only bites under a specific enabling condition: TP with tp_comm_overlap (TransformerEngine Userbuffers), fp8/mixed-precision on Hopper/Ada, PP>1 for microbatch-split thresholds, or multi-rank distributed collectives.
- A newly-added knob is set (or newly defaults) to a value that inserts an extra per-iteration collective / sync point (all_reduce, cuda.synchronize, all_gather_object) or resizes concurrency, changing latency/throughput without changing results.

**历史量级样本**：
- ~15-20x cuda-graph capture speedup (from in-code comment on the pre-existing GC-freeze optimization, not a measured regression)

**检测时该查什么（父快照核验）**：
- At the parent snapshot, diff the argparse default/action for any changed flag: did store_true→store_false (or dest= rename) flip the effective default for jobs that never pass the flag?
- Is the toggled feature on a hot path (per-iteration train loop, per-microbatch forward, per-collective, per-checkpoint-save)? Trace where the flag lands (bias_activation_fusion, ub_overlap_*, fp8_autocast, sync interval).
- Is the knob default-enabled and does it add an unconditional cross-rank collective or CPU-GPU sync (all_reduce/barrier/synchronize/all_gather_object)? Cost scales with rank count / DP size.
- For config→dataclass auto-copy: does the target field default True with no matching CLI arg name, making the toggle unreachable from CLI?
- Are newly-added config params actually forwarded from the builder call site into the context/engine constructor, or do they silently fall back to defaults?
- Check the enabling precondition (TE version, torch version, fp8-capable HW, PP>1, tp_comm_overlap) — the regression may only appear when that precondition holds.

**高发文件 top5**：`megatron/training/arguments.py (arg defaults / store_true↔store_false flips, new CLI knobs)`, `megatron/training/arguments.py: core_transformer_config_from_args (args→TransformerConfig mapping)`, `megatron/core/transformer/ (TransformerBlock forward, TE Linear/LayerNormLinear fusion & fp8_autocast paths)`, `megatron/core/pipeline_parallel/schedules.py (autocast/microbatch-split, per-step forward)`, `megatron/core/dist_checkpointing/ (FullyParallelSaveStrategyWrapper, cached distribution) and megatron/core/parallel_state / initialize_model_parallel (env-var/collective toggles)`

### failure-detection-gap  （13 张 train 卡）

**典型反模式**：
- Blocking wait with no timeout on a cross-process synchronization primitive (event.wait(), process.join(), socket.recv(), collective handshake) — if the peer/subprocess dies, the waiter hangs forever instead of failing fast.
- Scheduler admission checks that return a plain bool ('can_allocate'/'can_swap_in') and are consumed as `if not can_X(): break`; a request that can NEVER fit (KV blocks exceed total GPU blocks) is silently retried every pass, producing an unbounded stall with no error surfaced.
- Config combinations that make a request permanently unschedulable (e.g. per-item token budget larger than the max batch token budget) accepted at request time with no validation, only manifesting as an indefinite hang later.
- Health/readiness detection that relies on a single insufficient signal (e.g. `process.is_alive()` or in-process/same-process capability probe) that does not actually exercise the cross-process failure mode it is meant to guard, so failures go undetected until a downstream deadlock.
- No regression sentinel around throughput/recompilation/compile-time: a feature silently degrades (e.g. running a throughput feature on the engine that gives no overlap, or silent runtime recompilation) with no golden-value assertion or guard to catch it.

**显现条件 top3**：
- A subprocess/peer dies or fails during startup (engine spawn, checkpoint subprocess, RPC/DP handshake) and the parent blocks indefinitely on an untimed wait — no error, just a hang.
- A request/config that can never be satisfied (KV blocks > total GPU blocks; per-item mm tokens > max batched tokens; beam search n>1 on small block budget) enters the scheduler and stalls forever because the admission check only signals 'not now' rather than 'never'.
- A feature runs in an unsupported/mismatched context (sync engine vs async, middleware consuming request.receive(), buggy cross-process P2P, TPU XLA recompilation) and silently produces degraded throughput/util or a deadlock with no detection.

**历史量级样本**：
- Readiness-probe hang: indefinite; context.destroy() wait bounded at 30 seconds per added code comment
- UVM allocator load_inline: _compile_timeout(30s) SIGALRM guard added against otherwise-indefinite compile hang
- Dynamic-inference throughput golden values seeded (e.g. 18.42, 144.48, 7.129 tok/s across TP configs)

**检测时该查什么（父快照核验）**：
- Does any wait on an event/process/socket/collective lack a timeout? Trace what happens if the counterpart crashes — is there a fail-fast path or does it hang forever?
- For scheduler admission predicates: distinguish 'cannot fit right now' from 'can never fit'. If a request's requirement exceeds a hard total (total GPU blocks, max batched tokens), is there a fast-fail error instead of an infinite retry/break loop?
- Is the liveness/readiness signal sufficient? Does `is_alive()`/exitcode/in-process probe actually detect the cross-process failure, or is a stronger signal (poll timeout, exitcode check, cross-process IPC test) needed?
- Is the config validated at admission for combinations that make requests permanently unschedulable (per-item budget vs global budget, disable_chunked_mm_input)?
- Is there a golden-value / guard sentinel on throughput, compile/recompilation count, and startup time to catch silent degradation, and is it default-enabled or opt-in (env-gated, skipped under enforce_eager)?
- Does the feature assume a specific engine/frontend/hardware context (async vs sync engine, middleware presence, working P2P, torch_xla) that, if violated, degrades or deadlocks silently?

**高发文件 top5**：`vllm/core/**/block_manager* and scheduler.py (admission checks: can_allocate/can_swap_in)`, `vllm/engine/**/multiprocessing/* and entrypoints/openai/* (MQLLMEngine startup, zeromq readiness probe, API server retry loops)`, `vllm/config*.py / scheduler config validation (disable_chunked_mm_input, max_num_batched_tokens vs mm-item budget, DP compute_hash)`, `vllm/distributed/**/custom_all_reduce* and device P2P capability checks (_can_p2p / gpu_p2p_access_check)`, `checkpoint/decoupled checkpoint engine + cpp_extension load_inline paths (DeepSpeed DecoupledCheckpointEngine, Megatron UVM allocator compile timeout)`

### profiling-accounting-fix  （32 张 train 卡）

**典型反模式**：
- Sizing the KV cache from a memory snapshot DELTA taken at the wrong moment (e.g. `peak = init_free - end_free` at end of profile_run) instead of the true torch allocation peak; conflates torch/non-torch memory and mis-sizes the cache.
- Computing non-torch/baseline memory with an arithmetic that conflates categories, e.g. `non_torch = free_pre_profile - mem_get_info()[0]` or `total_allocated = mem_get_info()[1]-mem_get_info()[0]`, so NCCL/other-process buffers get double-counted or dropped.
- Taking the memory baseline snapshot BEFORE an allocating step runs (e.g. init_snapshot recorded before NCCL communicator init, or LoRA/sampler/encoder-cache setup running outside the DeviceMemoryProfiler block), so their memory is unattributed.
- Under-profiling the dummy worst-case run: skipping the sampler/encoder/second pooling task, caching raw dummy encoder outputs smaller than the runtime budget, or building the dummy sequence with the wrong token count (`max(seq_len,total_len)`, `max_num_batched_tokens//max_num_seqs`) — either over- or under-allocating.
- Using an allocator metric that only counts part of usage, e.g. `max_memory_allocated`/`allocated_bytes` instead of `memory_reserved`/`max_memory_reserved`, missing caching-allocator reserved bytes.
- Adding always-on instrumentation into the hot training/inference path (per-iteration `cuda.memory_stats()` writes, unconditional l2-norm all_reduce, timing barriers on the PP group) defaulted ON.

**显现条件 top3**：
- Multi-GPU / TP / EP / NCCL configs where non-torch (communicator) allocations are large and get mis-attributed, skewing available-KV-cache computation.
- Multimodal / MoE / pooling / LoRA models whose worst-case activation or workspace memory is under- or over-estimated during the dummy profile run (encoder cache, video frame caps, chunk-size workspaces, multiple pooling tasks).
- GPU shared with other processes or prior allocations at startup (init_gpu_memory < total), so baseline accounting errors surface as OOM or over-conservative KV sizing.

**历史量级样本**：
- available KV cache 8.2843 GiB -> 8.2923 GiB (test golden, ~8MB for opt-125m)
- VLLM_FUSED_MOE_CHUNK_SIZE workspace bound 64K -> 16K tokens
- video profiling frame cap _MAX_FRAMES_PER_VIDEO 600 -> 32 (num_frames)

**检测时该查什么（父快照核验）**：
- Is the profiling memory metric the true peak of the caching allocator (reserved/max_reserved), or a possibly-misleading snapshot delta / allocated-only value?
- Are all memory-consuming setup steps (NCCL init, LoRA manager, sampler, encoder/vision cache) executed INSIDE the profiler window and AFTER the baseline snapshot?
- Does the dummy worst-case profile run exercise every path that can peak memory (all pooling tasks, encoder + scatter buffer sizing, correct dummy seq length vs max_model_len)?
- Is the non-torch/baseline term computed without conflating torch vs non-torch or double-counting other-process memory?
- Is any added accounting/instrumentation on the per-iteration hot path or PP/model-parallel collective path, and is it default-enabled?
- Confirm whether a barrier/timer added for measurement can deadlock or stall (e.g. guarded on microbatch counts that go unmet).

**高发文件 top5**：`vllm/**/gpu_worker.py (determine_available_memory / init_device / MemorySnapshot)`, `vllm/worker/**/*worker*.py (determine_num_available_blocks, memory_profiling)`, `vllm/v1/worker/gpu_model_runner.py (profile_run / _dummy_pooler_run)`, `vllm/platforms/**/*.py (get_current_memory_usage) & multimodal profiling helpers (qwen2_vl.py, get_num_frames_with_most_features)`, `megatron/training/training.py & schedules (training_log memory stats, calc_params_l2_norm, 1F1B timing barriers)`

### wrong-index-dtype-hang  （2 张 train 卡）

**典型反模式**：
- Refactor drops a defensive dtype-reconciliation line before an elementwise op (e.g. removing `residual = residual if residual.dtype == x.dtype else residual.to(x.dtype)`), so mixed fp32/fp16 operands silently up-cast the result. In pipeline-parallel setups this desyncs the dtype of tensors sent/received across stages, and mismatched send/recv buffers deadlock.
- Cross-stage communication buffers assume a fixed dtype but the producing op emits a promoted dtype (fp32) due to operand mismatch, leaving one rank's collective/point-to-point op waiting forever.
- Spin-wait synchronization on a monotonically-incrementing signed integer flag using `while (*flag < reduce_id)`; once the counter wraps past INT_MAX the signed comparison mis-orders producer/consumer ids and a completed op is read as unfinished, hanging the kernel.
- Sequence/generation counters for collective ops typed as signed int with no wrap-around handling, so long-running jobs (~2B ops) trigger overflow-driven ordering bugs.

**显现条件 top3**：
- Pipeline-parallel training under AMP O1 mixed precision where residual is fp32 but the bias+dropout output is fp16, causing cross-stage dtype mismatch.
- Tensor-parallel with userbuffers comm-GEMM overlap using fused reduce-scatter/all-gather kernels on NVIDIA GPUs.
- Very long-running jobs whose global collective-op counter approaches/exceeds INT_MAX (~2B operations).

**历史量级样本**：
- hang (n/a)
- hang (n/a)

**检测时该查什么（父快照核验）**：
- Was a dtype-reconciliation / cast-guard line removed in a refactor on the hot residual/elementwise path?
- Do communication buffers (send/recv, reduce-scatter/all-gather) assume a fixed dtype while an upstream op can promote fp16->fp32?
- Is any spin-wait or ordering flag typed as signed int without wrap-around handling?
- Is the symptom a silent hang (no crash, no error) rather than a slowdown — i.e. one rank/kernel waiting indefinitely?
- Does it only reproduce under a specific parallelism mode (pipeline-parallel, or TP with comm-GEMM overlap) rather than single-GPU?

**高发文件 top5**：`megatron/ (transformer/residual bias-dropout-add paths)`, `TransformerEngine userbuffers CUDA kernel sources (comm-GEMM overlap)`, `pipeline-parallel send/recv communication modules`, `mixed-precision / AMP wrapper code`


## 大类：host-overhead

### custom-op-dispatch-overhead  （18 张 train 卡）

**典型反模式**：
- Wrapping a large forward region — including linear projections (GEMMs), gated RMSNorm, gates, and elementwise ops — inside a single opaque custom op (torch.ops.vllm.*, torch.library.custom_op), so torch.compile cannot see or fuse the projections and each launch pays a dispatch boundary.
- Registering hot-path ops via @torch.library.custom_op / torch.library.custom_op("ns::name", ...) instead of the lower-overhead direct_register_custom_op path; the custom_op decorator constructs extra machinery per call and adds significant dispatch overhead on small kernels.
- Calling a tiny op through a Python wrapper indirection (a function in _custom_ops.py that merely forwards to torch.ops._C.*, or a CustomOp subclass overriding forward() so every call re-enters the dispatcher self._forward_method(...)) instead of caching the op handle once and calling it directly.
- Using convenience tensor ops that dispatch into heavier composite kernels on the hot path (e.g. F.pad(x,(1,0)) to prepend a zero instead of torch.cat([x.new_zeros(1), x])) or tensor-subclass paths whose __torch_dispatch__ falls through to a default that silently dequantizes (FP8->high-precision) on record_stream/storage access.
- Doing per-forward quantization (scaled_fp8_quant) inside each backend's forward as a custom op that torch.compile cannot fuse, or using the apply()/autograd-tracked quantizer path (internal=False) on inference forwards where the direct internal=True storage-class path has less overhead.

**显现条件 top3**：
- torch.compile / CUDA-graph (piecewise) or XLA compilation is enabled, and the opaque custom-op boundary blocks fusion/memory-planning of the GEMMs and elementwise ops it wraps.
- Hot decode/forward path with many small ops per step (LoRA punica kernels, activation ops, RoPE, attention, all-reduce, quantization), where per-op dispatch/wrapper overhead dominates the actual kernel time.
- FP8 / block-quantized weight or KV paths and tensor-subclass quantized tensors, where dispatch fall-through triggers extra dequantize/quantize work or defeats compiler fusion.

**历史量级样本**：
- FP8 in-backend query quant: the total time these ops take is about the same as the whole attention kernel (issue-measured)

**检测时该查什么（父快照核验）**：
- Is this on the per-step decode/forward hot path, or a rare init path? (dispatch overhead only matters when the op is called frequently and the kernel is small relative to launch cost.)
- Are heavy sub-ops (linear projections/GEMMs, RMSNorm, gates) hidden inside a single opaque custom op that torch.compile / XLA cannot see into? Check whether projections could be hoisted into the module forward().
- How is the custom op registered — @torch.library.custom_op vs direct_register_custom_op? Prefer the direct helper to avoid dispatch construction overhead.
- Does a CustomOp subclass override forward() directly (re-entering the dispatcher each call), or does the caller go through a thin Python wrapper instead of a cached op handle (self.op = torch.ops._C.*)?
- Is torch.compile / piecewise CUDA graph actually enabled? Projection-decoupling optimizations are no-ops (or regressions) without it — check whether the change was later reverted for lack of compile.
- Platform sensitivity: does the custom-op boundary help on CUDA but hurt out-of-tree/XLA backends (TPU OOM, HPU forward overrides)? Guard with current_platform checks.
- For quantized paths: does the quantizer use the autograd apply() path (internal=False) or a tensor-subclass whose __torch_dispatch__ falls through to a default that dequantizes on record_stream/storage access?

**高发文件 top5**：`vllm/model_executor/layers/mamba/** and linear-attention mixers (mamba2, gdn, kda) forward paths`, `vllm/model_executor/layers/*custom_op* / _custom_ops.py and CustomOp dispatch base (activation, rotary, fused_moe)`, `vllm/lora/ops/**/punica sgmv/bgmv Triton kernels and their custom-op registration/libentry wrappers`, `vllm/attention/backends/** and layer.py (FlashInfer/Triton FP8 query quant, MLA rotary, unified_flash_attention/all-reduce/fused_experts op registration)`, `TransformerEngine transformer_engine/pytorch/module/linear.py & quantized tensor subclasses (Float8BlockwiseQTensor); DeepSpeed op_builder UtilsBuilder call sites`

### eager-error-string-build  （3 张 train 卡）

**典型反模式**：
- Passing an eagerly-evaluated expensive expression as an argument to a gated logging call, e.g. `logger.debug(obj.model_dump_json())` or `logger.debug(expensive_serialize(x))` — Python evaluates the argument before the logger checks its level, so the cost is paid even when DEBUG is disabled.
- Using f-string interpolation in log statements on hot paths, e.g. `logger.debug(f"append slot for {seq_group}")` — the f-string (including `__repr__`/`__str__` of complex objects) is built unconditionally regardless of log level.
- Building a helper/constant only needed on the error branch at the top of a frequently-called function, e.g. `SUPPORTED = [e.value for e in SomeEnum]` computed on every call but referenced solely inside a rare `else: raise ValueError(...)` message.
- Constructing error/diagnostic strings unconditionally at function entry rather than lazily inside the failure branch that actually consumes them.

**显现条件 top3**：
- Per-request or per-step invocation on CPU-side request handling / scheduling loops (e.g. request entrypoint, add_seq_group, per-seq-group decode scheduling), amplifying the redundant work.
- Repeated invocation of a function whose only expensive-string use is on the error path, e.g. weight loading / repeated weight reload calling the loader many times.
- Logging level set above DEBUG in production, so the eagerly-built argument or f-string is computed and immediately discarded.

**检测时该查什么（父快照核验）**：
- Is the call on a hot path (per-request, per-token, per-scheduler-step, or per-weight-load)?
- Is an expensive expression (serialization, model_dump_json, repr of large objects) passed as an argument to a level-gated logger call so it evaluates before the level check?
- Are f-strings used in logger.debug/info instead of the lazy `logger.debug("...%s", arg)` form?
- Is a constant/list-comprehension/error-string built at function entry but only used in a rare error/raise branch — could it be moved into that branch or module-level?
- Would the work be entirely wasted when logging is disabled or when no error occurs?

**高发文件 top5**：`vllm/entrypoints/**/serving_*.py (frontend serving paths, e.g. Anthropic/OpenAI messages)`, `vllm/entrypoints/**/api_server.py`, `vllm/core/scheduler.py (add_seq_group and decode scheduling loops)`, `vllm/model_executor/layers/fused_moe/**.py (FusedMoE weight_loader)`

### one-time-setup-on-hot-path  （37 张 train 卡）

**典型反模式**：
- Calling `packaging.version.Version(importlib.metadata.version(pkg))` (or similar package-metadata/version parsing) inside a per-forward/per-call method instead of computing it once at import/`__init__` and caching it on the object or a module global.
- Placing a `try: import ...; except ImportError` (or any module import that hits import-machinery / entry_points scanning) inside a per-layer/per-request hot function, so the import lookup runs on every microbatch/request rather than being hoisted to module scope.
- Rebuilding derived collections/tensors from scratch every iteration (e.g. filtering `parameters()` by `requires_grad`, recomputing `input_ids == some_id` masks, or allocating stride/scale tensors via `torch.full(...)`) instead of precomputing once in `__init__` and reusing the cached value.
- Reconstructing expensive objects on every invocation — a fresh `InferenceParams`/KV-cache, a oneDNN/CUTLASS/dnnl matmul `primitive_desc`, a multimodal processor, or a Jinja2 chat-template compile — inside the request/forward path instead of memoizing or warming up at init.
- Deferring JIT/AOT compilation (Numba `@jit`, DeepGEMM/librosa/numba kernels, C++ `load_inline`) into the first real request's execution path instead of a dedicated warmup at construction/model-init.
- Reading env vars per call via `std::getenv()`+`std::atoi()`/string construction in kernel-dispatch code, or eagerly allocating a collective/rendezvous buffer unconditionally in a shared init path that every job runs.

**显现条件 top3**：
- Setup work (version parse, import, object/primitive construction, tensor alloc, scale compute) sits on a per-forward / per-decode-step / per-microbatch loop, so cost multiplies by layers × iterations × requests.
- Setup work sits on a per-request path of an online serving endpoint (OpenAI-compatible API, logits-processor validation, generation-config lookup) and is paid on every incoming request.
- First-request / cold-start compile or handshake (Numba/librosa/DeepGEMM JIT, Jinja2 template compile, NIXL ZMQ handshake) runs inside execute_model / generate instead of a warmup, spiking first-request latency or stalling the model-runner step.

**历史量级样本**：
- first-request latency ~7s (librosa/numba JIT) + ~2.5s (multimodal input processor) [vllm transcription]
- n-gram proposer Numba JIT warmup: "usually takes less than 1 second" (moved to init) [vllm]

**检测时该查什么（父快照核验）**：
- Is this code on a hot path? Trace the caller: does it run per-forward, per-decode-step, per-microbatch, or per-request rather than once? Check function names like `forward`, `apply`, `build`/`_plan`, `__call__`, `create_*_completion`, `execute_model`.
- Does the body do work whose result is invariant across calls for the object's lifetime (version string, import result, mask/stride/scale tensor, compiled template, processor object)? If so it should be hoisted to `__init__`/module scope/cache.
- Are there implicit expensive operations hidden behind simple-looking calls — `importlib.metadata.version()`, `entry_points()`, `getenv`+parse, `torch.full`, scalar-to-device assignment forcing pageable HtoD, or a JIT trigger on first call?
- Is any compile/JIT/handshake being lazily triggered inside the execution path? Confirm there is a dedicated warmup at construction or model-init instead.
- Is the setup call unconditional / default-enabled in a shared init path (e.g. MPU init, module import) so it affects all configs including those that never use the feature?
- Does the setup allocate a collective/rendezvous or cross-node resource (symmetric-memory buffer, NIXL agent) that adds a synchronization/round-trip on top of the local cost?

**高发文件 top5**：`Megatron-LM: megatron/core/parallel_state.py (initialize_model_parallel / MPU init path)`, `Megatron-LM: megatron/core/transformer/**/attention.py & transformer_engine.py (TE wrappers, QKV split, version checks)`, `vllm: vllm/v1/attention/backends/*.py (FlashInfer / FlashAttention / MLA metadata builders — build/_plan hoisting)`, `vllm: vllm/model_executor/layers/fused_moe/**.py (CUTLASS/DeepGEMM/oneDNN MoE forward paths, per-call alloc/primitive/scale)`, `vllm: vllm/entrypoints/openai/serving_*.py (per-request preprocessing, chat-template warmup, generation-config lookup, logits-processor validation)`

### quadratic-python-structure  （37 张 train 卡）

**典型反模式**：
- Linear scan of a growing collection on every hot-path step: `for c in candidates: if c.matches(args): ...` or `x in some_list` / `min(enumerate(sizes), key=...)` re-run per forward/decode step, turning per-step O(n) into O(n^2) over the run.
- Rebuilding a whole container each step to remove/add one element: `self.running = [r for r in self.running if r not in stopped]` or `deque(sorted(seq_groups, key=...))` executed unconditionally every scheduling step, allocating O(n) even when only one item changed.
- Recomputing over the entire growing sequence per token: hashing `hash(tuple(input_ids))`, scanning `end_token in input_ids`, or concatenating `prompt_ids + output_ids` to slice/tuple it on every block-hash / decode step (O(n) per step -> O(n^2)).
- Allocating many small GC-tracked Python objects on the per-step / capture hot path: `list[list[int]]`, `list[dict[int, Logprob]]`, nested `dict[hash, dict[id, block]]`, or per-item dataclass objects, inflating cyclic-GC traversal cost as the live-object set grows.
- Using auto-generated dataclass `__eq__` for identity/linked-list comparisons (`block == self.free_list_head`) so equality recurses over all fields (including neighbor pointers) instead of an O(1) identity check.
- Repeated `list.index(...)` / `while`-loop rescans inside a token-rewriting loop (`new_tokens = prefix + repl + postfix; new_tokens.index(...)`), an O(n^2) rebuild over total tokens.

**显现条件 top3**：
- Per-step hot paths (per decode step, per scheduling step, per forward) where the scanned/rebuilt/allocated structure grows with sequence length, batch size, or number of concurrent requests (e.g. running queue 1K+ reqs).
- Long sequences / long prompts combined with per-token work (reasoning models with think/end tokens, prefix caching with ~15k-token prompts, large top_logprobs), yielding O(n^2) accumulation over generation.
- Startup / CUDA-graph-capture or checkpoint-save paths where many long-lived objects accumulate and Python's cyclic GC repeatedly rescans the growing live heap, or where per-item bin/index assignment is a linear scan.

**历史量级样本**：
- CUDA-graph capture GC freeze: ~15-20x faster capture (in-code comment)
- Multimodal no-match string prompt (10000 chars): 'was taking seconds' -> '< 100ms'
- Frontend beam search: processing time reduced by nearly 40% (issue-measured)
- list[list[int]] -> list[np.ndarray] for sampled_token_ids: 19% throughput boost, infinite request rate, facebook-125m (issue-measured)

**检测时该查什么（父快照核验）**：
- Is this code on a per-step hot path (per-token decode, per scheduling step, per forward), or on startup/capture/checkpoint? Confirm the caller frequency at the parent snapshot.
- Does a loop/scan/comprehension/sort run over a collection whose size grows with sequence length, batch size, cached-block count, or concurrent-request count? If so, is it re-run each step (O(n) per step -> O(n^2))?
- Is the structure default-enabled on common paths (prefix caching, structured/guided decoding, reasoning parsers, beam search) or gated behind a rare flag?
- Are many small GC-tracked objects (nested lists/dicts, per-item dataclasses) allocated per step or per capture, inflating cyclic-GC cost? Could a flat/columnar structure or gc.freeze reduce it?
- Are equality comparisons using dataclass `__eq__` where identity (`is`) is intended, causing deep recursive field comparison?
- Could the per-item linear assignment (min-bin, index lookup, membership test) be replaced by a heap, precomputed map, ordered structure, or O(1) arithmetic?

**高发文件 top5**：`vllm/core/scheduler.py, vllm/v1/core/sched/*.py (per-step running-queue rebuild, output processing)`, `vllm/core/block/**, vllm/v1/core/**/block_pool.py, evictor*.py (prefix-cache lookup, LRU eviction, free-block queue, CoW tracking)`, `vllm/**/reasoning/*.py, structured_output/*.py, guided/outlines backends (per-token is_reasoning_end / hashing / should_advance)`, `megatron/core/**/cuda_graphs.py, CudaGraphManager (runner selection scan, GC-freeze during capture)`, `megatron/core/datasets/**/helpers.cpp + dataset.py, distributed checkpoint _split_by_size_and_type (sample-index build, writer-bin assignment)`

### redundant-hot-path-work  （201 张 train 卡）

**典型反模式**：
- Computing a heavy op (log_softmax/softmax, casting to fp32) over the FULL tensor (all sequence positions × full vocab) and only afterwards indexing/selecting the rows actually needed — do the selection first, then the expensive math.
- Rebuilding derived Python objects per iteration/per-forward purely to read one element: list comprehensions of slice views over a range, dataclasses.asdict() deep-copies of param dataclasses, deep .clone()/copy of 'expensive' config objects on every call.
- Doing expensive setup unconditionally before the guard that would skip it: string/stack-walking work (inspect-based call-path formatting) before an `if not enabled` check; recording interprocess CUDA events / opening files / creating buffers before checking whether any request/replica actually needs them.
- Materializing O(n^2) or full-length artifacts regardless of need: dense triangular causal masks per sample/microbatch, padding sequences to a fixed configured max length instead of the batch's actual max, zeroing multi-GB KV buffers on reset, wrapping inference-only models in DDP (gradient/all-reduce buffers).
- Recomputing identical results every step instead of memoizing/caching: per-token get_added_vocab() dict rebuilds, per-group metadata builder runs that differ only in one field, missing @lru_cache on per-seq-length masks, not passing is_first_microbatch so TE re-casts FP8 weights every microbatch.
- Running the same computation twice: index_select immediately overwritten by gather; attention run once only to fill KV cache then again for the real output; count_zeros() + blocking all_reduce/.item() every step when not needed.

**显现条件 top3**：
- Cost scales with a large dimension that is mostly unused: vocab_size × non-selected positions, seq_len² masks, data_parallel_world_size × num_buckets, fixed max_seq_length ≫ batch's actual max.
- Work sits on a per-iteration / per-microbatch / per-forward / per-token hot path and is default-enabled (or unconditionally executed) regardless of whether the guarding feature is active.
- Feature/backend makes the redundant artifact unnecessary: TE attention builds its own mask, decode-only path needs only the last token, inference-only eval doesn't need DDP, disabled profiling doesn't need the message string.

**检测时该查什么（父快照核验）**：
- Is this on a hot path (per-token / per-microbatch / per-iteration / per-forward / per-attention-layer)? Multiply cost by call frequency.
- Is the expensive work done BEFORE the guard/early-return that could skip it? Move the `if not enabled` / replica / need-check above the setup.
- Do we compute over the full tensor then index down, when we could select/slice first? Look for log_softmax/softmax/gather over full vocab or full sequence.
- Is a result identical across calls or differing in only one field? Candidate for @lru_cache / memoization keyed by the varying inputs.
- Is a heavy object deep-copied (asdict, clone, list-of-slices) just to read/index one field? Precompute or shallow-copy.
- Is a full-size buffer/mask/pad allocated or zeroed when the batch's actual size is smaller, or when the backend builds its own? Size to actual, or delete.
- Is the same op executed twice (e.g. index_select then gather, attention twice)? Remove the dead first computation.
- Is a config option (is_first_microbatch, weight caching, average_in_collective) left at a default that forces redundant recompute/rescale?

**高发文件 top5**：`Megatron-LM: megatron/core/inference/**/dynamic_inference_context.py (decode/prefill bookkeeping, log_probs, KV buffer)`, `Megatron-LM: megatron/core/transformer/**/attention.py, custom_layers/transformer_engine.py, fused_softmax.py (masks, GQA expand, TE forward, is_first_microbatch)`, `Megatron-LM: megatron/core/distributed/**/param_and_grad_buffer.py, optimizer/*.py, clip_grads.py (shard_buffer, grad scaling, count_zeros)`, `Megatron-LM: megatron/core/transformer/moe/*.py (token_permutation, aux-loss); data loaders megatron/**/dataset_helpers.py, gpt_dataset.py, get_batch (mask build/broadcast, padding)`, `vllm: vllm/**/sampling_params.py, model tokenizers (deepseek_v32), attention metadata builders, model_executor/models/gemma3.py, kv connector (LMCache event/recording)`


## 大类：inference-serving

### async-output-overlap  （20 张 train 卡）

**典型反模式**：
- Blocking GPU→CPU sync each step to materialize sampled tokens (e.g. an explicit `self._sync_device()` / `.cpu()` on the sampled-token tensor) directly in the model-runner output path, so the D→H copy cannot overlap with the next forward step. Missing async output path means a plain `ModelRunnerOutput` is returned synchronously instead of an async wrapper backed by a torch.cuda.Stream/Event.
- Strict schedule-then-consume alternation in the batch-queue engine step: returning immediately after scheduling without consuming any ready output, forcing lock-step and leaving GPU bubbles between iterations.
- Consumer-side draining of per-request output queues one item per loop iteration (yield one RequestOutput per queue item), letting an asyncio.Queue accumulate multiple deltas under load instead of coalescing them on the producer side into a single slot.
- Advancing `num_computed_tokens` only inside update_from_output (after the model executes) rather than at schedule time, which serializes bookkeeping with execution and breaks pipelined/chunked-prefill overlap.
- Device/feature gates that hard-disallow overlap: whitelists that only permit certain device_type/executor (e.g. cuda-only), or mutual-exclusion flags like `allow_async_output_proc = use_async_output_proc and not is_multi_step`, silently disabling overlap for whole classes of configs.
- Streaming HTTP responses declared with a buffering media_type (e.g. application/json) instead of an incremental SSE/stream type, so intermediaries buffer the whole body and the first token chunk is never flushed early.

**显现条件 top3**：
- Async scheduling enabled (use_async_scheduling / max_concurrent_batches>1 / batch_queue_size>1) so CPU scheduling is meant to overlap GPU execution; a residual blocking D→H sync or synchronous output return re-serializes the two.
- Multi-step scheduling (num_scheduler_steps>1) or streaming online serving (RequestOutputKind.DELTA) where output pythonization/detokenization must overlap the next forward and outputs must be flushed/coalesced promptly.
- Pipeline-parallel / data-parallel or non-CUDA backends (TPU, XPU, HPU, uniproc executor) where the overlap path is gated off or not wired, so those configs regress to strict lock-step.

**历史量级样本**：
- Fuse Pad with MXFP8-Quantize (~1% perf gain); Fuse MoE Finalize with Slice (~1% perf gain); RoPE+Q+CacheUpdate fusion (~2% perf gain); special BF16 gemm for router gemm (1-2% perf gain) (issue-measured, adjacent optimization card)

**检测时该查什么（父快照核验）**：
- Is there a blocking device→host sync (`_sync_device`, `.cpu()`, cuda synchronize, or 'GPU -> CPU Sync happens here') on the sampled-token/pooler output in the per-step model-runner path? Could it be deferred behind an AsyncGPUModelRunnerOutput / torch.cuda.Stream+Event instead?
- At the engine-step / batch-queue caller: does scheduling and output-consumption alternate strictly, or does the step consume ready outputs while also scheduling the next batch?
- Is the overlap feature default-enabled and reachable for this config, or gated by a device whitelist / mutual-exclusion flag (multi-step vs async output proc, PP>1, spec-decode, Ray SPMD, uniproc)?
- On the async streaming path: is the per-request queue drained/coalesced on the producer side into a single slot, or does the consumer yield one item at a time and accumulate backlog under load?
- Is num_computed_tokens (and similar per-request bookkeeping) advanced at schedule time to allow pipelining, rather than only after model execution?
- For streaming HTTP endpoints, is the media_type an incremental/SSE type that flushes the first token early, not a buffered application/json body?

**高发文件 top5**：`vllm/v1/core/sched/ (async_scheduler.py, scheduler.py) — schedule/consume and num_computed_tokens advance`, `vllm/v1/worker/*model_runner*.py and gpu/tpu/xpu runners — execute_model, async_callback, AsyncGPUModelRunnerOutput, sync points`, `vllm/v1/engine/ (core.py step_with_batch_queue, async_llm.py generate/output collector) — RequestOutputCollector coalescing`, `vllm/config.py & vllm/engine/arg_utils.py — device_type/executor gates, verify_async_output_proc, async/multi-step defaults`, `vllm/engine/*llm_engine*.py / output_processor.py — async output processing, SchedulerContext, _process_model_outputs, stream_interval`

### encoder-execution-mode  （25 张 train 卡）

**典型反模式**：
- Recomputing a fixed encoder output on every autoregressive decode step: forward() unconditionally runs the (image/audio/text) encoder each call instead of caching it behind an `if encoder_hidden_states is None:` / inference-cache guard.
- TP-sharding a vision/ViT encoder by default (QKV/Row/Column/MergedColumnParallelLinear) so every encoder layer pays an all-reduce/all-gather collective, when a replicated data-parallel path (ReplicatedLinear + per-rank batch shard) would avoid the per-layer collective.
- Running encoder work on ranks/paths that never consume its output: `_execute_mm_encoder`/`_gather_mm_embeddings` executed on every PP rank though embeddings are only used on one PP stage; placing the projector in the per-forward merge path instead of the cached encoder path.
- Always instantiating and executing the full encoder stack even when unused: loading vision-encoder weights for text-only serving, or running all encoder layers when only up to `vision_feature_layer` is needed; serializing encoder inputs one-per-step instead of batching (`assert len(encoder_outputs) in (0,1)`).
- Leaving encoder torch.compile / compute-budget knobs (compile_mm_encoder, max_num_encoder_input_tokens, encoder_cache_size) as hardcoded/flipped defaults so the encoder silently runs eager or with mismatched budgets.

**显现条件 top3**：
- Multi-GPU tensor-parallel serving (world_size>1, benefit cited at TP=8) of multimodal vision encoders where the encoder runs TP-sharded rather than data-parallel — no effect at TP=1.
- Autoregressive / decode-phase inference on encoder-decoder or multimodal models (T5, Whisper, LLaVA, Qwen*-VL) where a fixed encoder output is recomputed per step or per non-consuming rank.
- Batched / concurrent multimodal or audio requests, or text-only serving of a VL model, where encoder batching, encoder skipping, or layer truncation is not applied.

**历史量级样本**：
- ~5% throughput / ~7% latency (ViT wrapped-op + torch.compile path on H100, per docstring; not a measured migration number)
- ~10% throughput at TP=8 for batch-level DP mm encoder (per docs/configuration/optimization.md)
- up to +40% throughput for Conv3D-based vision encoders under mm_encoder_tp_mode=data (per optimization.md)

**检测时该查什么（父快照核验）**：
- Is the encoder forward on a hot path? Check whether encoder output is recomputed every decode step vs. cached behind an `encoder_hidden_states is None` / inference-cache guard.
- Who runs the encoder? Verify encoder execution is gated to ranks/PP-stages that actually consume the embeddings, not run unconditionally on every PP rank.
- Is the encoder TP-sharded by default? Look for QKV/Row/Column/MergedColumnParallelLinear in ViT layers incurring per-layer all-reduce; check whether mm_encoder_tp_mode/use_data_parallel/disable_tp offers a replicated DP alternative.
- Is the full encoder always instantiated/executed? Check for unused vision weights loaded in text-only mode, encoder layers beyond vision_feature_layer, and one-input-per-step serialization vs. batched encoder inputs.
- Are encoder knobs at sane defaults? Inspect compile_mm_encoder, max_num_encoder_input_tokens, encoder_cache_size, and mm_encoder_tp_mode defaults for eager/placeholder values on the target platform/model.

**高发文件 top5**：`vllm/model_executor/models/*vl*.py, *vision*.py (Qwen2-VL, Qwen2.5-VL, Qwen3-VL, GLM-4.x-V, InternVL, Kimi-VL, Step3, Idefics2/MiniCPM-V, Llama4, DotsOCR vision encoders)`, `vllm/model_executor/layers/linear.py (LinearBase / Column/Row/QKV/Merged ParallelLinear, disable_tp / ReplicatedLinear plumbing)`, `vllm/attention/layer.py and vision attention wrappers (MultiHeadAttention -> MMEncoder/vit_attn_wrappers path)`, `vllm/config.py / CompilationConfig / MultiModalConfig (compile_mm_encoder, mm_encoder_tp_mode, encoder compute budget), vllm/v1 scheduler encoder-cache sizing`, `megatron/core/models/T5 & multimodal (LLaVA) forward paths (encoder_hidden_states / image_tokens_count guards)`

### kv-cache-capacity  （34 张 train 卡）

**典型反模式**：
- Computing KV-cache block/page byte size with a hardcoded factor of `2 *` (separate K and V) without a `use_mla` branch — for latent-attention (MLA) models that store only one compressed vector, this doubles the per-block estimate and halves allocatable blocks (also seen on CPU backends duplicating value_cache_block = key_cache_block).
- Sizing per-rank KV buffers from the FULL model layer count (`num_layers=model_config.num_layers`) under pipeline parallelism, so each PP stage that holds only a subset of layers over-allocates by a factor of pp_size; likewise dividing a total buffer independently per rank with unequal layer allocation produces divergent block counts across ranks.
- Preallocating the KV cache to a fixed/model-derived max sequence length (e.g. a default max_seq_length or an uncapped model-config max) rather than the actual prompt/sequence length, over-committing memory or degrading latency when max_model_len is left unspecified.
- Deriving num_blocks from a base sizing path and then logging an override (e.g. num_gpu_blocks_override) or fudging padding (head_size // padded_head_size) without actually assigning the corrected value back — the intended cap/override silently no-ops.
- Building the KV-cache spec with the activation/weight dtype (torch.get_default_dtype()) instead of the configured kv-cache dtype, so an fp8 KV cache is sized as if it were bf16; or profiling peak (transient) memory as steady-state footprint, under-sizing the cache.

**显现条件 top3**：
- Pipeline parallelism (pp_size>1), especially with uneven layer allocation across stages, during dynamic/continuous-batching inference — per-rank block counts diverge or over-allocate by pp_size.
- MLA / multi-latent-attention models (e.g. DeepSeek-V2/V3) where a `2*` K+V accounting assumption is applied to a single-latent cache, halving allocatable blocks (GPU V1 page_size_bytes and CPU cache_block_size).
- Default/unspecified capacity knobs: max_model_len or max_sequence_length defaulting to a large model-derived value, num_preallocate_tokens headroom, or per-backend defaults (CPU 4 GiB, Neuron one-block-per-seq) that mis-size the buffer.

**历史量级样本**：
- step_count 22 (max_requests=None) vs 34 (max_requests=4) from added test_max_requests unit test
- cross-layer KV sharing: 2x max context length / 2x KV blocks for same memory (e.g. 655360 vs 327680 blocks; 2 * 1310720)
- KV cache preallocated to max_seq_length default 2560 vs actual prompt/seq length
- MLA `2*` mis-accounting: ~2x KV-cache memory over-counted, halving allocatable blocks
- PP over-allocation factor equals pp_size
- num_preallocate_tokens default 64 (~4 extra blocks at block_size 16 per request)

**检测时该查什么（父快照核验）**：
- Is the per-block/per-page byte size derived with a fixed `2 *` (K and V) factor? Check whether a `use_mla` (single-latent) branch exists and whether CPU/GPU paths agree.
- Under PP>1, is the KV buffer sized from the full num_layers or from this rank's local layer count? Are stages guaranteed equal layer allocation, and is the total-buffer division consistent across ranks?
- Where does the cache get its sequence budget — actual prompt/seq length, or a model-derived/default max (max_model_len, max_sequence_length)? Is that default capped?
- For override/padding knobs (num_gpu_blocks_override, head_size alignment): is the corrected value actually assigned back, or only logged/computed then discarded?
- Does the KV-cache spec use the configured kv-cache dtype (e.g. fp8) rather than the model activation dtype? Is available memory computed from steady-state (not transient peak) footprint?
- Is this a default-enabled path (continuous batching, hybrid/sliding-window/cross-layer sharing managers) that changes how many blocks are allocatable per request?

**高发文件 top5**：`vllm/v1/core/kv_cache_utils.py (get_num_blocks / spec sizing)`, `vllm/v1/core/*kv_cache_manager* and sliding-window / hybrid / chunked-local managers`, `vllm/attention/backends/* and vllm/platforms/* (page_size_bytes, get_kv_cache_shape, per-backend memory: CPU/TPU/Neuron)`, `megatron/**/inference/**/dynamic_context (DynamicInferenceContext block/buffer sizing)`, `megatron/**/inference/**/forward_step / inference_params (max_sequence_length preallocation)`

### prefill-decode-kernel-split  （20 张 train 卡）

**典型反模式**：
- Routing an entire mixed prefill+decode batch through the heavy prefill kernel whenever any prefill token is present (e.g. `if has_prefill:` / `if num_prefills>0:` gates), so single-token decode rows get processed by the query-length-tiled prefill kernel instead of a decode-specialized path.
- Unifying prefill and decode into one general kernel that tiles over query length (BLOCK_M) regardless of query_len==1, making the decode-heavy / long-context case pay the general-kernel cost (observed slowdown vs. the split-path predecessor).
- Prefill-only or scoring requests (num_tokens_to_generate==0) stepping the generation loop token-by-token up to max_prompt_length instead of computing all prompt positions in a single prefill pass.
- Gathering the entire context KV into fresh contiguous tensors (`torch.empty((total_tokens, H_KV, D))`) per layer before the attention call, turning a chunked-prefill/extend workload memory-bandwidth bound.
- Hard threshold constants that steer 'small prefills' through the decode/MQA pathway (e.g. reorder_batch_threshold) set without regard to the actual query-length distribution, mis-routing mid-size prefills to the wrong kernel.

**显现条件 top3**：
- Mixed prefill+decode batches under continuous batching / chunked prefill, where decode rows are query_len==1 and prefill rows are long.
- Long-context / chunked-prefill (extend) workloads and prompts approaching or exceeding max_num_batched_tokens / max_tokens.
- Specific backend or hardware paths (Triton/FlashInfer/FlashMLA/AITER-ROCm/TPU-Pallas) where the prefill-vs-decode kernel dispatch or a routing threshold was recently changed or unified.

**历史量级样本**：
- Triton unified vs split-path attention: old ~600ms -> new ~1500ms for an 8k-token sequence (issue-measured)

**检测时该查什么（父快照核验）**：
- Is this on the attention/mixer hot path executed every layer per step? A wrong kernel choice multiplies across layers and steps.
- At the routing gate, does presence of any prefill token force the whole batch (including decode rows) through the prefill kernel? Check the `if has_prefill / num_prefills>0` condition.
- Is there a magic threshold (e.g. reorder_batch_threshold=512, decode_query_len==1) that decides prefill-vs-decode routing? Was its default changed?
- Does the change unify previously-separate prefill/decode kernels, and is the unified kernel tiled over query length (penalizing query_len==1 decode)?
- Is a new feature/env flag default-on or default-off (e.g. VLLM_V1_USE_PREFILL_DECODE_ATTENTION, --kv-sharing-fast-prefill, enable_chunked_prefill)? Verify which path the default now takes.
- Does the prefill path allocate/gather full-context contiguous KV tensors per layer (memory-bandwidth bound)?
- For prefill-only/scoring requests, does the code loop token-by-token instead of one batched prefill?
- Which backend/hardware does the path apply to, and does high-TP / low-heads-per-rank fall back to a different (slower) kernel?

**高发文件 top5**：`vllm/v1/attention/backends/*.py (triton_attn, flash_attn, flashinfer, flashmla, rocm/aiter, tpu)`, `vllm/attention/ops/* (chunked_prefill_paged_decode, unified_attention, causal_conv1d, selective_scan)`, `vllm/model_executor/layers/mamba/* and models/{jamba,mamba,gemma3n}.py (mixer forward, prefill/decode split)`, `vllm/core/scheduler.py + scheduling budget (chunked-prefill scheduling, token_chunk_size)`, `megatron/inference/**/dynamic_inference_engine / controller (chunked prefill, prefill-only stepping)`

### prefix-state-caching  （33 张 train 卡）

**典型反模式**：
- Force-disabling prefix caching via a config guard when a feature is enabled (e.g. `self.cache_config.enable_prefix_caching = False` under full_cuda_graph, prompt_embeds, sliding_window, or on a specific platform/backend), silently forcing full-prefix recompute even when reuse would be safe.
- Backends whose prefix-prefill path is `raise NotImplementedError` (or missing), so a prefix that is already computed/cached is recomputed every step because there is no read path from cached KV blocks.
- Cache membership/lookup that does not update recency or key correctly: probing an LRUCache with `in`/`__contains__` (no LRU bump), keying an encoder/mm cache by request_id instead of a content hash, or embedding an ephemeral non-content id (e.g. lora_int_id) into the block hash so equivalent content never matches.
- Block-hash / cache-eligibility arithmetic that goes out of sync with a companion config: computing cached-token counts or block hashes with a raw block_size while another component scaled it (DCP world size, hybrid group block_size), or subtracting the full spec_token_ids length so accepted tokens are excluded from caching.
- Deciding a block is a shared common-prefix block via `ref_cnt == num_running_requests` while async/disaggregated KV transfer holds extra refs, undercounting reusable prefix blocks.

**显现条件 top3**：
- Prefix caching (APC / enable_prefix_caching) enabled together with a feature that historically disabled it or lacked a read path: hybrid/mamba/sliding-window models, multimodal/vision-language inputs, prompt embeds, full_cuda_graph, or non-CUDA backends (TPU, HPU).
- Workloads with shared/repeated prompt prefixes or repeated multimodal inputs across concurrent/sequential requests — the exact case where reuse should hit but a keying/guard bug causes recompute.
- Interaction with parallelism/decoding features that rescale or add refs: Decode Context Parallel, TP+MLA (KV not sharded), async/disaggregated KV transfer, speculative decoding, and multi-engine/data-parallel LoRA with inconsistent adapter ids.

**检测时该查什么（父快照核验）**：
- At parent snapshot, grep for any `enable_prefix_caching = False` / `enable_caching = False` force-disable guards and check whether the gating feature is now actually compatible (full_cuda_graph, sliding_window, prompt_embeds, TPU/HPU platform checks).
- Check the block-hash inputs: does block_size match any scaled value (dcp_world_size, hybrid group block_size)? Does the hash include ephemeral non-content ids (lora_int_id) or a within-block offset that breaks matching across requests?
- Verify cache lookups bump LRU recency (not bare `in`/`__contains__`) and are keyed by content hash (mm_hash/blake3) rather than request_id.
- For each backend/hardware path, confirm the prefix-prefill / forward_prefix code path exists (not NotImplementedError) so cached prefix KV is read rather than recomputed.
- Check common-prefix / cached-token counting under async KV transfer and speculative decoding: `ref_cnt == num_running_requests` assumptions and `num_computed + num_new - len(spec_token_ids)` arithmetic against token-budget clamping.
- Is this on the hot scheduling/allocate_slots path and is the feature default-enabled? Silent throughput/latency loss (no crash) is the dominant symptom.

**高发文件 top5**：`vllm/v1/core/kv_cache_manager.py and vllm/v1/core/kv_cache_*.py (allocate_slots, block hashing, common-prefix logic)`, `vllm/v1/core/*block_hash* / generate_block_hash_extra_keys (BlockHashType, mm/lora extra keys)`, `vllm/config*.py (CacheConfig._verify_args, check_and_update_config, platform force-disable guards)`, `vllm/**/attention backend prefix-prefill paths (Pallas/TPU, HPU PagedAttention, FlashAttention cascade)`, `vllm/v1/**/mm/encoder cache & multimodal preprocessor cache (EncoderCacheManager, MirroredProcessingCache/MMHasher)`

### sparse-layer-update  （4 张 train 卡）

**典型反模式**：
- Zero-token / empty-batch shortcut in an expert or MLP forward that returns a passthrough of the input (e.g. `output = input`) instead of routing through the weight matmuls — the weight Parameters then receive no gradient, and any per-parameter grad-reduce/reduce-scatter hook tied to that Parameter never fires.
- Building optimizer partitions and master/optimizer states (fp32 master copy, Adam momentum+variance) over ALL params in `optimizer.param_groups` without filtering `requires_grad=False`, so frozen parameters get full optimizer-state memory allocated for them.
- Registering per-parameter backward hooks that only fire when the parameter is touched in forward; under MoE + data-parallel, ranks that don't route to a given expert never activate its hook, so the collective (reduce-scatter/all-reduce) is unbalanced across ranks and deadlocks.
- Opt-in sparse layer-update schedules (progressive layer drop) that skip Transformer blocks per step without ensuring the parameter/gradient-sync contract still holds when a block is dropped.

**显现条件 top3**：
- Expert/MoE parallelism (EP>1 or ZeRO-3 + data-parallel) where different ranks activate different experts, combined with grad-sync mechanisms (--overlap-grad-reduce, --use-distributed-optimizer, or ZeRO-3 reduce-scatter hooks).
- Presence of a code path that skips weight application: empty/zero-token batches, or frozen (requires_grad=False) parameters, or dropped layers.
- bf16/fp16 training with distributed optimizer + fp32 master copies and Adam states over the full param group.

**历史量级样本**：
- 2.5X faster pretraining / 24% faster (progressive layer drop, from docs)

**检测时该查什么（父快照核验）**：
- Does any forward branch (zero-token, empty-batch, dropped-layer) return the input unchanged, bypassing the weight matmuls? If so, those Parameters get no gradient and any hook keyed on them will never fire.
- Are grad-reduce / reduce-scatter hooks registered per-parameter and fired only when the parameter is used in forward? Under MoE + DP, will every rank activate the same set of parameters each step? If not, collectives can hang.
- Does optimizer/master-state construction iterate over the full `param_groups` without filtering `requires_grad=False`? Check for memory blowup from frozen params.
- Is the sparse-update feature default-enabled or opt-in? Check whether the default path changes grad-sync behavior.
- Is this on the training hot path across all DP/EP ranks (collective participation must be symmetric across ranks each step)?

**高发文件 top5**：`megatron/core/transformer/moe/ (GroupedMLP / grouped-gemm expert forward)`, `megatron/core/distributed/ (grad-reduce hooks, buckets, distributed optimizer)`, `deepspeed/runtime/zero/ (ZeRO-3 partitioning, reduce-scatter hooks, param-group init)`, `deepspeed/runtime/zero/*optimizer* (fp16/fp32 partition + Adam state construction, trainable-parameter-group filtering)`, `deepspeed/runtime/progressive_layer_drop* / model layer forward (opt-in sparse block update)`

### speculative-decoding  （18 张 train 卡）

**典型反模式**：
- Hardcoding query/sequence length to 1 in the decode attention call (e.g. `max_seqlen_qo = 1`, `q = q.unsqueeze(1)`, `q_len_per_request = 1`), so each speculated draft/verify token is processed as an independent single-token decode instead of a uniform multi-token batch — the attention kernel loses the batched-spec fast path.
- Scoring proposals via 'batch expansion': expanding each sequence into k+1 single-token decode requests, so the target forward-pass cost scales with batch_size × proposal_len instead of using an MQA/uniform-spec scorer.
- Running draft-token proposal synchronously inline in the engine step / EngineCore busy loop before schedule()/execute_model(), so the step blocks on draft generation and CPU-GPU sync before returning outputs (missing async/overlapped drafting).
- Always computing and serializing full logprobs tensors (dense (batch, sample_len, vocab) allocations) from GPU to CPU each spec step, even when accepted tokens differ from target-sampled tokens — needless device-to-host sync and memory churn.
- Auto-disable-by-batch-size logic that is unreachable: routing to the cheap no-spec path only on `num_lookahead_slots == 0` or empty batch and ignoring the computed `disable_all_speculation` flag under high load.
- Token-budget caps that count speculated/placeholder tokens against a request's max_tokens/max_model_len, partially truncating uniform draft-token batches per step.
- Disabling MTP/spec-decode entirely under specific parallel/backend configs via blanket asserts (e.g. forcing `reorder_batch_threshold == 1`), or dropping the bonus (k+1) token for KV-cache draft models to avoid KV bookkeeping.

**显现条件 top3**：
- MLA / spec-decode attention backends (ROCm AITER MLA, FlashInfer-MLA trtllm-gen, FA3 MLA) where query length > 1 for MTP but the kernel is fed q_len=1 — throughput loss with no functional error.
- Uniform-decode spec batches where a token-budget cap or per-request placeholder accounting truncates draft tokens, or where draft-token counts must stay uniform across requests in a step (async scheduling / spec decode).
- High running-queue / large-batch regimes where speculation should auto-disable (`running_queue_size >= disable_by_batch_size`) but the disable flag is ignored, or where batch-expansion scoring inflates compute.

**检测时该查什么（父快照核验）**：
- Is the draft/verify query length hardcoded to 1 anywhere on the decode attention path (unsqueeze(1), max_seqlen_qo=1, q_len_per_request=1)? Confirm the backend actually supports uniform-spec / multi-token decode.
- Is draft-token proposal on the hot path — run inline/synchronously in execute_model or the EngineCore busy loop before scheduling — instead of overlapped/async in the model runner?
- Do scored proposals use batch expansion (cost ~ batch × proposal_len) rather than an MQA / uniform-spec scorer?
- Are logprobs unconditionally computed and serialized GPU→CPU every step, with dense (batch, sample_len, vocab) allocations, even when unused?
- Is any auto-disable/knob (disable_by_batch_size, disable_all_speculation) actually consulted on the routing branch, or dead-code bypassed by a narrower condition?
- Do token-budget/max_total_tokens caps count speculative placeholder tokens, risking partial truncation of a uniform draft batch?
- Are MTP/spec paths blanket-disabled for a parallelism/backend combo (asserts forcing threshold==1, DCP+MLA, bonus token dropped for KV-cache drafts)?

**高发文件 top5**：`vllm/spec_decode/ (spec_decode_worker, mqa_scorer, ngram_proposer, multi_step_worker, top1_proposer)`, `vllm/v1/worker/gpu_model_runner.py (and gpu_worker.py) — drafting / propose_draft_token_ids, spec_token_ids`, `vllm/v1/attention/backends/ (mla/*, flashinfer_mla, flash_attn) — supports_uniform_spec / reorder_batch_threshold, q reshaping`, `vllm/v1/core/sched/ (scheduler) — max_total_tokens / num_output_placeholders token-budget caps`, `vllm/v1/core/... engine step / EngineCore loop (propose_tokens, execute_model routing)`


## 大类：io-startup

### async-io-overlap  （5 张 train 卡）

**典型反模式**：
- Performing checkpoint/state disk writes inline on the training main thread, blocking forward/backward compute instead of overlapping I/O with computation (synchronous `save(...)` on the critical path).
- Doing per-request/per-sample preprocessing (deserialization, cache lookup, structured-output/grammar init, decode+dtype conversion) inline in the add-request or data-fetch hot path rather than in a worker/background stage.
- Decoding input records and doing numpy→tensor conversion on the main thread with no bounded prefetch queue, so data prep serializes with the training step.
- Blocking HBM→host / host→NVMe tensor copies before every persist, without double-buffering (fill/drain) to overlap copy with the async write.
- Missing an opt-in async path (default is fully synchronous), or forgetting the `torch.cuda.synchronize` / non-blocking copy semantics needed to make overlap safe.

**显现条件 top3**：
- Distributed/data-parallel checkpointing where per-rank checkpoint size is large — synchronous save cost scales with checkpoint volume and stalls the step.
- Heavy per-request or per-sample preprocessing (multimodal cache lookup, deserialization, grammar/structured-output init, TFRecord decode) done inline, most impactful under high request rate or large batch.
- CUDA/GPU workloads writing to local NVMe/durable storage where HBM→host→disk copy latency is high and could otherwise be hidden behind compute (e.g. ZeRO stage-1, pipeline scheduler).

**历史量级样本**：
- DeepSpeed DataStates-LLM: up to 48x faster checkpointing, 2.2x faster end-to-end training
- DeepSpeed FastPersist: over 20X (Phi-3-Mini on 8xGen5 NVMe)
- DeepSpeed ZeRO-Inference: 7→17→26 tokens/sec (4xGen4→4xGen5→8xGen5 GDS)
- DeepSpeed I/O: reads 10→27→48 GB/s, writes 5→11→26 GB/s

**检测时该查什么（父快照核验）**：
- Is the disk/NVMe write or state serialization on the training critical path (blocking the step) vs. offloaded to a forked process / background thread / async caller?
- Is there an opt-in async flag (e.g. async_sharded_save / async checkpointing engine) — and is the synchronous path still the default (regression risk when async not enabled)?
- For copies before I/O: are GPU→host / host→NVMe copies non-blocking and double-buffered (fill/drain overlap), or synchronous?
- Is preprocessing/data-loading executed inline in the add-request or fetch hot path, or moved to a worker with a bounded prefetch queue (num_workers>0, daemon loader thread)?
- Is a required CUDA sync (torch.cuda.synchronize before fork) present so the overlapped write reads consistent tensors?
- Caller frequency: per-step / per-request hot path multiplies the inline cost.

**高发文件 top5**：`Megatron-LM: dist_checkpointing/ (async save, DistributedAsyncCaller / schedule_async_call)`, `Megatron-LM: data loading / BERT dataset paths (threaded TFRecord loader, MultiprocessLoader)`, `vllm: v1/engine/core.py (EngineCore.add_request / preprocess_add_request), request construction & mm_input_cache`, `DeepSpeed: async checkpointing engine (DataStates-LLM), fast_file_writer / Double_IO_Buffer (DeepNVMe/FastPersist)`, `DeepSpeed: docs/_tutorials/*async-checkpointing* and NVMe/GDS I/O offload modules`

### process-launch-overhead  （6 张 train 卡）

**典型反模式**：
- Using the `spawn` multiprocessing start method for worker/helper processes whose parent has already imported a heavy stack (torch, engine modules), forcing every child to re-import and re-initialize the entire interpreter state on each launch.
- Spawning fresh processes inside a nested loop over device pairs (O(N^2) or N*N*2 subprocesses) at startup to run per-pair probes (e.g. P2P access checks via a new process group), with no caching or opt-out for trusted environments.
- Forking a large memory-intensive parent process on every operation (e.g. per-checkpoint-save) so that copy-on-write pages and duplicated allocations create host-memory pressure/OOM.
- Not preloading heavy modules into a forkserver control process, so each forked worker re-imports the module transitively instead of importing it once and inheriting it.

**显现条件 top3**：
- Linux hosts using the fork start method (incompatible with spawn-only platforms).
- CUDA GPU workloads where tensors are owned by the parent process (fork requires GC/ownership guards).
- Environments with trusted/working P2P drivers where the startup P2P probe is redundant.

**检测时该查什么（父快照核验）**：
- Is process creation on a hot/startup path or repeated per-operation (per-save, per-device-pair) rather than one-time?
- What start method is used (spawn vs fork vs forkserver)? spawn re-imports the full stack per child; check whether the parent already holds heavy imports that could be inherited.
- Does the launch count scale with world_size or num_dev (e.g. nested N*N loops)? Is there caching or an opt-out env var for trusted hardware?
- Is a large/memory-heavy parent being forked repeatedly (COW page duplication, CUDA tensors owned by parent)? Check for _disable_gc / memory-pressure guards.
- For forkserver: are heavy modules preloaded via set_forkserver_preload + ensure_running so they are imported once in the control process?
- Is the chosen start method platform-portable (fork is Linux-only; spawn-only platforms will break)?

**高发文件 top5**：`vllm/**/executor/*multiproc* (multiprocessing executor / worker spinup)`, `vllm/distributed/**/*p2p* (P2P access check / device capability probing)`, `vllm/v1/engine/async_llm.py and api_server entrypoints (forkserver preload)`, `megatron/**/dist_checkpointing/**/async* (TemporalAsyncCaller, async checkpoint save path)`, `**/envs.py or config modules defining VLLM_WORKER_MULTIPROC_METHOD / VLLM_SKIP_P2P_CHECK`

### redundant-load-startup-work  （61 张 train 卡）

**典型反模式**：
- Unconditionally running weight initialization at module construction even when weights will be immediately overwritten by a checkpoint load (e.g. always passing a real init_method / calling `_initialize_affine_weight_*` regardless of a `perform_initialization`/from-checkpoint flag).
- Building a full unpartitioned master tensor of shape (output_size, input_size) on every process/rank, then splitting/scattering — instead of directly initializing only the local shard.
- Every rank in a data-parallel/replication group redundantly reading the same replicated shards or full state dict (ShardedObjects, non-sharded state) from disk on checkpoint load, rather than distributing reads across the group and broadcasting.
- Eagerly materializing large buffers with a writing initializer (torch.full/torch.zeros/torch.empty) that touches every page (KV cache, dummy optimizer exp_avg/exp_avg_sq), instead of lazy/deferred or reuse-based allocation.
- Eager top-level imports of all submodules in a package `__init__.py`, pulling transitive deps at import time instead of lazy module-attribute resolution.
- Recomputing invariant work inside a per-expert / per-tensor / per-size loop (permutation layouts, pin_memory, model-class inspection) instead of caching and reusing across iterations.
- Constructing full submodules (embedding tables, vision towers) on every pipeline/parallel stage regardless of whether that stage/config actually uses them (missing `if pre_process:` / feature gate).

**显现条件 top3**：
- Distributed / multi-rank setups where the same load or build work is replicated across data-parallel, tensor-parallel, or virtual-pipeline ranks; cost scales with group/world size and number of dataset file prefixes.
- Startup/warmup paths that build datasets, load distributed checkpoints, or capture CUDA graphs — where per-rank filesystem stress and per-size/per-expert repetition dominate wall time.
- Configurations where constructed work is provably unused: loading from checkpoint (init discarded), text-only serving of multimodal models (vision tower unused), non-first pipeline stages (embedding unused).

**历史量级样本**：
- DataLoader init 'can take several minutes' opening/memory-mapping many dataset files (readme-reported)
- Subprocess-based model inspection adds ~4s on local SSD, easily 2x+ on network filesystem (issue-measured)

**检测时该查什么（父快照核验）**：
- Is this init/allocation/load on the startup or warmup hot path (module __init__, process_weights_after_loading, dataloader/dataset build, checkpoint load, cudagraph capture)?
- Is the work being done on every rank redundantly? Check whether it is gated by is_dataset_built_on_rank / pre_process / parallelization-group rank, and whether VPP `ignore_virtual` flags accidentally widen the set of ranks doing the work.
- Will the result be immediately overwritten (weights loaded from checkpoint) or is it default-enabled but unused (perform_initialization=True by default, vision tower with image limit==0)?
- Does a large buffer use a writing initializer that faults in every page, when a lazy/non-materializing or reuse path would suffice?
- Is invariant work (imports, permutation layouts, pin_memory, model-class inspection, attr-doc parsing) recomputed inside a loop or on every call instead of cached / lazily resolved?
- Is the cost filesystem-bound (memory-mapping many files, subprocess spawn, from_pretrained fallback) and thus amplified on shared/network FS?

**高发文件 top5**：`megatron/core/datasets/** (blended/megatron dataset build, dataloader init)`, `megatron/core/dist_checkpointing/** (fully-parallel load strategy, dist-opt state load)`, `megatron/core/tensor_parallel/layers.py & mpu/layers.py (affine weight init)`, `vllm/model_executor/models/** and model registry/loader (lazy model inspection, model-class resolution)`, `vllm/model_executor/layers/quantization/** (MoE process_weights_after_loading, per-expert shuffle)`

### serial-io  （8 张 train 卡）

**典型反模式**：
- Iterating over a collection of I/O work items (files, dataset prefixes, per-worker RPCs) in a plain `for` loop and doing each blocking read/stream/build sequentially, instead of submitting them to a ThreadPoolExecutor / batched call (`for f in files: stream_file(f)` vs `stream_files(files)`).
- Funneling distributed state (optimizer shards, checkpoint files) through a single privileged rank (`if rank != 0: return` / gather-scatter via DP-rank-0), serializing what could be partitioned across all ranks (e.g. `partition_uniform(num_layers, dp_size)`).
- Issuing per-item blocking remote calls inside loops or comparators (e.g. `ray.get(worker.get_node_ip.remote())` per worker and again per sort comparison) instead of batching all remote lookups into one collective fetch.
- Hardcoding concurrency knobs to serial defaults (reader threads = 1, ninja/parallel build disabled, no `--threads` for compilation) so I/O and compilation cannot exploit available CPU/storage parallelism.

**检测时该查什么（父快照核验）**：
- Is this on a startup / weight-load / checkpoint / build hot path that runs once but blocks readiness?
- Does a `for` loop perform blocking I/O (stream/read/write/build) per item where items are independent and could be parallelized or batched?
- Is distributed state serialized through a single rank (rank-0 gather/scatter, `if rank!=0: return`) instead of partitioned across ranks?
- Are there per-item blocking remote/RPC calls inside loops or comparators that could be collected into one batched call?
- Are concurrency defaults set to serial (readers=1, ninja disabled, no compile `--threads`) such that the default-enabled config leaves parallelism unused?
- Does the benefit scale with a countable dimension (num files/prefixes/workers/DP ranks/CPU cores)?

**高发文件 top5**：`*/model_executor/model_loader/*weight*.py, tensorizer/runai streamer loaders (vllm)`, `*/executor/ray_*executor.py — Ray worker creation & IP-sort logic (vllm)`, `*/data/gpt_dataset* / blended dataset index-building modules (Megatron-LM)`, `*/dist_checkpointing/optimizer & distributed optimizer sharding paths (Megatron-LM)`, `setup.py / builder & op_builder (nvcc/ninja build flags), pipeline checkpoint save modules (DeepSpeed)`

### slow-fs-path  （4 张 train 卡）

**典型反模式**：
- Per-element read path issues a fresh open()+seek()+readinto() syscall on every __getitem__/get() call (unbuffered, buffering=0) instead of using a memory-mapped or OS-page-cache-backed path, so hot dataloading incurs one or more syscalls per sample.
- Flipping a storage-access config default (e.g. an mmap flag) from the cheap/cached path to the per-call syscall path, silently switching all callers who rely on the default onto slow I/O.
- Staging GPU tensor I/O through a pinned CPU bounce buffer for every transfer instead of a direct GPU<->storage path, adding an extra copy on the hot storage path.
- Placing a cache/autotune/temp directory on a network filesystem (NFS) by default, so writes/flushes (often at atexit) cross the network and stall or hang.

**显现条件 top3**：
- Disk/filesystem I/O-bound dataloading with high sample throughput on slow storage.
- Default config selects per-call syscall read path over mmap/page-cache path.
- Cache/autotune/temp directory located on an NFS mount and flushed at exit.

**检测时该查什么（父快照核验）**：
- Is this read/write on the per-sample hot path (called once per __getitem__/get/element)? Count syscalls per element.
- Which access path does the default select at parent snapshot — mmap/page-cache vs fresh open+seek+readinto(buffering=0)?
- Did a config default flip (e.g. mmap False->True or vice versa) change the effective I/O path for existing callers?
- For GPU offload paths: does the transfer go through a pinned CPU bounce buffer or a direct GPU-storage path?
- Is any default cache/autotune/temp dir potentially on NFS or a slow mount, especially if flushed at exit?

**高发文件 top5**：`Megatron-LM: megatron/core/datasets/indexed_dataset.py (IndexedDataset _getitem_mmap / _getitem_file)`, `Megatron-LM: megatron/core/datasets/blended_megatron_dataset_config.py (mmap_bin_files default)`, `DeepSpeed: csrc/aio / DeepNVMe AIO op descriptors (io_op_desc_t / cpu_op_desc_t / GDS)`, `DeepSpeed: triton cache dir resolution (_default_cache_dir / is_nfs_path)`, `*/datasets/*, */data*/*, */aio/* storage read-path modules`

### weight-load-strategy  （24 张 train 卡）

**典型反模式**：
- Eagerly materializing the full weight set into an intermediate container before handing it to the loader (e.g. building a `name -> tensor` dict for all params, or running multiple filter passes over the iterator with itertools.tee) — buffering doubles/triples peak host RAM because every consumed element is retained until all branches finish.
- Non-lazy checkpoint read: converting/loading the entire checkpoint eagerly (pickle-based torch.load of .bin, np.load without mmap_mode='r', or a numpy .npy disk round-trip per parameter) instead of memory-mapped/lazy slicing — full tensor materialized per rank on startup.
- Every worker/rank reading and re-sharding the full checkpoint (peak host RAM = sum over co-located workers), rather than sharding the read or serializing loads across local ranks so only the owning rank materializes its shard.
- Gating a fully-parallel / distributed load strategy on the wrong flag (reusing the save-side flag, leftover `# TODO: change to load` placeholders) or leaving a default that switches the I/O backend (zarr->torch_dist, CPU-vs-GPU init, safetensors auto-fallback) without accounting for the new backend's collective/startup cost.

**显现条件 top3**：
- Tensor-parallel and/or data-parallel with multiple workers/ranks co-located on one node: each rank reads or re-shards the full checkpoint, so peak host RAM and startup latency scale with the number of co-located ranks and model/checkpoint size.
- Large models / large-vocab embeddings / multi-shard checkpoints where a non-lazy or single-threaded read dominates startup time (disk/network-IO-bound loading).
- Distributed checkpoint load on multi-GPU/multi-node where the load strategy relies on CUDA collectives/all_gather across DP/CP groups and its benefit scales with group size (and can regress if shards concentrate in one group or the wrong flag is set).

**历史量级样本**：
- ~1GB re-consumed when draft model stopped sharing embed_tokens with target (Eagle/MTP, llama-3-class model)

**检测时该查什么（父快照核验）**：
- At the weight-loading entry point, is the full weight set buffered in an intermediate container (dict / itertools.tee / list) before load_weights, instead of streamed via a generator?
- Is the checkpoint read lazy/memory-mapped (safetensors safe_open/get_slice, np.load(mmap_mode='r')) or does it eagerly materialize the whole file (torch.load of .bin, np.load without mmap, per-param .npy round-trip)?
- Under TP/DP, does every co-located worker read the full checkpoint (peak host RAM = sum over ranks)? Check whether sharded/serialized loading is available and which flag gates it.
- For distributed/fully-parallel load strategies: is the wrapper gated on the correct load-side flag (not the save flag / a TODO placeholder), and does the default backend change (zarr vs torch_dist, CPU vs GPU init) alter startup collective cost?
- Is the fast/opt-in loader (fastsafetensors, tensorizer, sharded_state, multi-threaded, RunAI distributed streaming, eager read) default-off, and is the slow path still the default on the hot startup path?
- For spec-decode/draft models: are shared embedding weights (embed_tokens / lm_head) still shared vs re-loaded — a guard removal can silently re-consume the freed memory.

**高发文件 top5**：`vllm/model_executor/model_loader/ (weight_utils.py: *_weights_iterator, safetensors/tensorizer/fastsafetensors/sharded_state loaders)`, `vllm/model_executor/models/ (per-model load_weights: eagle/mtp draft models, pixtral, llama)`, `megatron/core/dist_checkpointing/strategies/ (torch.py, fully_parallel.py: FullyParallel{Save,Load}StrategyWrapper, distribute_shards_to_ranks)`, `megatron/**/data/*dataset* (samples-mapping index np.load / mmap_mode) and args/checkpoint arg parsing (--ckpt-fully-parallel-load, --dist-ckpt-format, --use-cpu-initialization)`, `deepspeed/runtime/engine.py and ZeRO stage-3 / AutoTP checkpoint loading paths (sharded / pipelined checkpoint load)`

### weight-transfer-sync  — ⏭️ 跳过（no train perf cards for this leaf）


## 大类：kernel-efficiency

### algorithmic-compute-reduction  （24 张 train 卡）

**典型反模式**：
- Computing an expensive reduction (softmax/log_softmax, full descending sort) over the entire tensor and only afterward slicing down to the rows/positions actually needed — e.g. running log_softmax over all sequence positions x vocab then indexing the last token, or full O(V log V) vocab sort when only top-k is requested.
- Iterating over the whole logical range (all KV tiles up to full seq_len, all experts via per-expert nonzero+scatter, every video frame decoded) when structural constraints (causal mask, sliding window, capacity cap, requested frames) mean most of that work is discarded — pay-then-drop instead of skip-upfront.
- Materializing a dense quadratic or full-grid intermediate (Q×KV attention mask, [tokens, experts, capacity] dispatch/combine einsum grid, dense [1,seq,seq] boolean mask, wide pre-absorbed weight matrices) that could be replaced by ragged/segmented/streaming computation.
- Using an asymptotically heavier primitive where a cheaper one suffices: O(n log n) argsort for an inverse permutation solvable by scatter in O(n); repeated full-context KMP scans (one per n-gram length) instead of a single pass; cryptographic hash (SHA-256) over full decoded buffers where a fast non-crypto hash or hashing compressed source suffices.

**显现条件 top3**：
- Long sequences / large vocab / high token counts where the wasted work scales with the dimension not needed — benefit grows with seq_len/window ratio, vocab size, number of experts x capacity, or number of vision/video tokens.
- Structural sparsity present but ignored: causal or sliding-window attention, packed/variable-length (THD) sequences with padding, MoE capacity limits, or top-k-only sampling where full-range work dominates.
- Prefill / decode hot paths and per-forward multimodal preprocessing (image/video decode, vision transformer permutations) invoked every step or every request.

**检测时该查什么（父快照核验）**：
- Is a reduction/sort/hash computed over the full tensor before an index/slice narrows it down? Move the narrowing before the expensive op.
- Does a loop or block-iteration cover the full range when a mask/window/capacity/causal structure guarantees most iterations are discarded? Bound the loop by the actual valid range.
- Is a dense quadratic (Q×KV) or full-grid intermediate materialized where ragged/segmented/cu_seqlens-based computation avoids padding compute?
- Is an O(n log n) primitive (argsort/full sort) used where O(n) scatter/topk would do, or a cryptographic hash where a fast hash over compressed bytes suffices?
- Is the optimization on a default-enabled hot path or opt-in (new config knob / backend / env var)? Many here are opt-in — check the default remains the old path.
- Does it sit in per-step decode, prefill, or per-request preprocessing (attention kernel, sampler, MoE dispatch, multimodal loader)?

**高发文件 top5**：`vllm/v1/sample/** and sampler top-k/top-p apply paths (apply_top_k_top_p)`, `vllm/**/attention/** unified/Triton/FlexAttention/MLA kernels and block-mask build paths`, `vllm/multimodal/** and vllm/assets/video.py (image hasher, video loader backends, vision transformers)`, `megatron/core/**/moe/** (SwitchMLP/token dispatcher, topk_softmax_with_capacity) and inference log-prob/generation paths`, `TransformerEngine attention.py / JAX fused_attn THD (packed-sequence seqlens/offsets, mask-to-seqlens)`

### correctness-forces-slow-path  （39 张 train 卡）

**典型反模式**：
- Introducing an opt-in correctness/determinism mode (batch-invariant, deterministic, reproducible) whose enabled branch swaps optimized fused kernels for reference/custom kernels, asserts away flash-attention, forces num_splits=1, or dequantizes low-precision weights to a wider dtype for a bitwise-stable matmul.
- Forcing autocast disabled / pinning matmuls to fp32 (e.g. `with torch.cuda.amp.autocast(enabled=False)`) around linear/GEMM/reduce paths for numerical stability, defeating Tensor Core mixed-precision throughput.
- Disabling a fast path with a blunt `if False:`, a hardcoded `self.use_fast = False`, or commenting out a `@support_torch_compile` / fast-kernel decorator as a quick correctness workaround, leaving the reference path permanently active.
- Broadly forcing `enforce_eager=True` / `use_cudagraph=False` / `use_inductor=False` for an entire model family or backend to dodge one correctness bug, disabling CUDA-graph capture and torch.compile across the board.
- Selecting a reference/iterative fallback (MATH SDP kernel, per-expert dense loop over all tokens, non-compiled block-mask builder, torch-compile bitmask apply) instead of the fused/vectorized kernel because the fast kernel produced wrong results or hung on some shape/arch.

**显现条件 top3**：
- A correctness/determinism/reproducibility mode or workaround flag is enabled, routing execution onto the reference path instead of the fused fast kernel.
- Specific hardware + dtype + shape combinations (e.g. Hopper/SM90/SM100/Blackwell, fp8/bf16, K>128, prob_n>256, head_size in a fixed set, weight dims divisible by 128) where the fast kernel was known-buggy and thus gated off.
- Feature interaction that disables a whole optimization stack: MLA disabling chunked-prefill+prefix-caching, multimodal disabling prefix caching, DP>1+MoE forcing eager, quantized model families forcing enforce_eager.

**历史量级样本**：
- ~5% throughput / ~7% token latency on H100 (Qwen2.5-VL vision blocks, from disabling the just-added torch.compile fast path)
- ~25% average GEMM speedup forfeited (Marlin 2:4 sparse prefill fast tiling disabled for correctness)

**检测时该查什么（父快照核验）**：
- Is the swapped-in path on a per-step hot path (every decode step, every GEMM, per-token, per-expert loop)? Bitmask apply / rope / attention / linear all run each step.
- Is the fast path disabled unconditionally (hardcoded False, `if False:`, commented decorator, forced enforce_eager) vs. narrowly gated to the actually-broken shape/arch?
- At parent snapshot, was the optimized kernel default-enabled and measured (docstring/PR claims like ~5% throughput, ~25% GEMM)? Check for speedup magnitudes now being forfeited.
- Does the correctness branch cascade into disabling CUDA graphs / torch.compile / inductor / autocast, multiplying the cost beyond the single kernel?
- Is the guard broader than the bug (whole model family, whole backend, all dtypes) when the correctness issue was specific (one quant mode, one arch, one shape)?
- Was the trade a temporary workaround pending an upstream fix (dependency version bump, root-cause patch)? Check if a later revert/unpin restores the fast path.

**高发文件 top5**：`vllm/model_executor/layers/quantization/** (Marlin/GPTQ/AWQ/CUTLASS/fp8 GEMM kernels + apply_*_linear paths)`, `vllm/attention/** and vllm/v1/attention/** (MLA, flash/flex/flashinfer backend selection, block-mask builders)`, `vllm/platforms/cuda.py / vllm/config.py (check_and_update_config, __post_init__ forcing enforce_eager/eager/prefix-caching/chunked-prefill)`, `megatron/core/tensor_parallel/layers.py and transformer/{attention.py,rope_utils.py} (autocast/fp32 forcing, deterministic/batch-invariant kernel swaps)`, `vllm/model_executor/layers/fused_moe/** (fused vs iterative/dense per-expert MoE, permute/unpermute kernels)`

### dma-access-order-relax  （7 张 train 卡）

**典型反模式**：
- Building a full concatenated/flattened buffer on the CPU (host `torch.cat(...).view(-1)`) before moving H2D, so the copy and the transfer are serialized on the wrong device instead of concatenating device-side after the transfer.
- Hardcoding a single memory-access path (e.g. `tl.make_block_ptr` + `tl.load`/`tl.store`, or `coalesce(...)` inner layouts) that ignores TMA/descriptor-friendly granularity, missing the wgmma/`make_tensor_descriptor` fast path on newer arch.
- Attaching aggressive cache/eviction hints (`cache_modifier='.cg'`, `eviction_policy='evict_last'`, CacheGlobal/CacheStreaming PTX policies) to loads of large reused weights, evicting data that will be re-read and thrashing L2.
- Emitting scoped memory-order PTX (`st.release.sys`/`ld.acquire.sys`, `fence.sc.gpu`) that is either stronger than the algorithm requires (adds barrier latency) or uses instructions unsupported by the minimum target arch (breaks/disables the fast path).

**显现条件 top3**：
- Multi-GPU collective/overlap paths (tensor-parallel broadcast, custom all-reduce, userbuffer comm+GEMM overlap, MoE expert layers) where memory access order and fences sit on the critical path.
- Arch-gated fast paths: TMA/cp.async/wgmma require Hopper/SM90 (or SM80+ for async copy); wrong or missing gating either falls back to a slow path or disables the feature on older arch (e.g. Pascal sm_60/61).
- Memory-bound GEMM / small-tensor shapes where transfer layout granularity and cache-hint eviction dominate over compute.

**检测时该查什么（父快照核验）**：
- Is a host-side concat/flatten preceding an H2D copy? Prefer transferring first, then concatenating on-device.
- Are cache modifiers / eviction policies applied to large, repeatedly-read operands (weights, expert matrices)? Verify they don't evict soon-reused data from L2.
- Are memory-order primitives (release/acquire/fence) actually required by the algorithm's ordering, or can a weaker scope be used?
- Does any PTX/descriptor path assume a compute capability higher than the minimum supported target? Confirm fallback path exists for pre-Hopper/Pascal/AMD.
- Is the inner tile/copy layout TMA-friendly (aligned granularity, e.g. 256-wide) rather than a generic coalesced layout?
- Is this in a hot collective path exercised under TP>1 / MoE / overlap so a per-chunk fence or extra copy is repeated many times?

**高发文件 top5**：`megatron/**/tensor_parallel/*.py (broadcast_data / TP data utilities)`, `vllm/**/attention/**/triton*.py and fused_moe/*kernel*.py (FLA / MoE Triton kernels)`, `vllm/**/quantization/machete/** (W4A8/W4A16 prepack layout copy)`, `csrc/**/custom_all_reduce* and distributed collective PTX barrier code`, `TransformerEngine/**/userbuffers/** and DeepSpeed inference CUDA kernel/mem_access headers`

### fused-backend-swap  （197 张 train 卡）

**典型反模式**：
- Overriding forward() to always delegate to the pure-PyTorch/reference path (e.g. forward_cuda just calls forward_native, or a fused-backend flag left commented out / hard-coded False) so the fused/optimized kernel is never reached.
- Inheriting a subclass kernel from the fast base but never overriding apply_weights, so it silently falls back to the base (e.g. CUTLASS/cuBLAS) path instead of the intended Triton/int8 kernel.
- Relying on @torch.jit.script / legacy JIT-fuser to fuse elementwise ops (bias+gelu, bias+dropout+add, swiglu) — becomes a no-op after the JIT fusion backend is deprecated, leaving unfused eager kernels.
- Computing an inherently elementwise/reduction result via a heavyweight op (per-token 1xN·Nx1 batched matmul, index_select vs gather, CPU FAISS flat index) instead of the fused/vectorized equivalent.
- Backend selection gate (use_flashinfer / use_aiter / --attention-backend / --transformer-impl / --use-flash-attn) defaulting to or falling back to the slow reference backend when the optimized library is importable but not detected.

**显现条件 top3**：
- A backend/impl selector defaults to (or silently falls back to) the reference path — e.g. transformer-impl=local, attention-backend unfused, forward_cuda→forward_native — while the fused library (TransformerEngine, FlashAttention/FA3, FlashInfer, AITER, DeepEP, CUTLASS/Triton) is actually available.
- Framework/library version change breaks the fusion assumption (PyTorch>=2.2 deprecating nvFuser in torch.jit; torch 2.9.x disabling CUDNN Conv3D), turning previously-fused ops into slow eager/unfused kernels.
- Hardware/dtype/shape gates for the fast kernel are not met so the op degrades to a general path (non-Hopper for FA3, missing FP8/bf16, head_dim>128, hidden_size outside persistent-LN table, kernel!=stride/padding!=0 for conv mulmat).

**历史量级样本**：
- "significant performance regression" (qualitative; Conv3D via CUDNN disabled in torch 2.9.x — no number given)

**检测时该查什么（父快照核验）**：
- Is the op on a hot per-step path (attention, MoE dispatch/combine, layernorm, cross-entropy over vocab, elementwise bias-fusion, patch-embedding conv)?
- At the parent snapshot, which backend does the selector resolve to by default, and does the caller pass the flag explicitly? Check whether forward_cuda/apply_weights actually reaches the fused kernel or just delegates to the native/base impl.
- Is a fused/optimized kernel default-enabled or opt-in? If opt-in, is the gate (HAVE_TE/HAVE_FA3/use_flashinfer/use_aiter/triton_kernels availability) satisfied in the target environment?
- Did a PyTorch / library / driver version bump silently disable a fusion backend (nvFuser JIT, CUDNN Conv3D) the code relies on?
- Are hardware/dtype/shape constraints (sm_90 for FP8/FA3, fp16/bf16 only, head_dim<=128, hidden_size in persistent-LN list, kernel==stride conv) met — otherwise does it fall back to the unfused general kernel?
- Is a fused kernel being replaced by an elementwise/vectorized equivalent, or vice versa (bmm vs elementwise mul, index_select vs gather, CPU vs GPU index)?

**高发文件 top5**：`megatron/core/transformer/**/ (transformer_layer specs, inference_layers.py, attention/flash paths)`, `megatron/core/transformer/moe/** (token dispatcher, fused_a2a.py, permutation, weighted swiglu)`, `megatron/core/fusions/** (fused_layer_norm, bias_gelu/bias_dropout, swiglu JIT fusion)`, `vllm/model_executor/layers/** (rotary_embedding, quantization scaled_mm/fp8, fused_moe backends, conv)`, `pretrain_*.py / initialize / arguments (transformer-impl, attention-backend, jit_fusion_options defaults)`

### kernel-config-tuning  （156 张 train 卡）

**典型反模式**：
- Missing device-and-shape-specific tuned kernel config file (e.g., fused-MoE Triton JSON keyed by E/N/dtype/device_name): when no matching file exists the kernel silently falls back to generic default/heuristic tile params (BLOCK_SIZE_M/N/K, GROUP_SIZE_M, num_warps, num_stages), which are usually suboptimal.
- Config-file lookup key mismatch: building the config filename from a truncated or non-canonical device string (e.g. 'H200' instead of 'NVIDIA_H200', or unhandled variants like 'NVIDIA_H200_NVL') so the tuned JSON is never found and the path degrades to defaults.
- Reusing a single launch/tiling config across kernels or shapes with different aspect ratios (e.g. one config passed to both shrink K=hidden,N=rank and expand K=rank,N=hidden GEMMs), or hardcoding a worst-case size hint (expected_m = max_num_tokens) that makes the autotuner pick a generic path.
- Hardcoded/overriding tile or dispatch constant that bypasses tuned defaults (e.g. returning a fixed TILE_SIZE for a special path, or missing a device-capability branch such as SM90 vs SM100 opt-flags, split_k, or swap-AB small-M configs).

**检测时该查什么（父快照核验）**：
- At parent snapshot, is there a matching tuned config JSON for the exact (E, N, dtype, block_shape, device_name) tuple? If absent, the fused_moe/GEMM path falls back to default heuristics.
- How is the config filename built (get_config_file_name / get_device_name)? Verify it uses the canonical full device string and handles family variants (e.g. *_NVL) — a truncated/mismatched key silently disables tuning.
- Is a single launch config reused across kernels/shapes with different aspect ratios, or is a size hint (expected_m, hint_override) hardcoded to worst-case? Check per-kernel/per-shape config selection.
- For small-M / decode shapes: are dedicated configs (swap-AB, split_k pinning, effective-rows rounding, tile-size overrides) present per device capability, or does one generic default cover all M buckets?
- Is a special-case path (e.g. multimodal/PrefixLM, LoRA MoE) overriding the tuned defaults with a hardcoded constant on a hot attention/GEMM path?

**高发文件 top5**：`vllm/model_executor/layers/fused_moe/configs/E=*,N=*,device_name=*.json`, `vllm/model_executor/layers/fused_moe/ (fused_moe.py, get_config_file_name / config lookup)`, `csrc/quantization/ and CUTLASS FP8 scaled_mm dispatch (sm90/sm100 fp8 config selection)`, `vllm/compilation/ (decorators.py, collective_fusion.py — Inductor/allreduce-fusion threshold tables)`, `vllm/model_executor/layers/fused_moe/ LoRA + DeepGEMM masked grouped-GEMM paths (batched_deep_gemm_moe)`

### kernel-fusion  （193 张 train 卡）

**典型反模式**：
- Elementwise activation chains executed as separate eager kernels (e.g. `torch.pow(F.relu(x), 2)`, chunk+silu+mul for GLU, bias-add then activation) instead of being wrapped in a JIT/torch.compile fuser or a fused autograd Function — leading to many small kernel launches and extra intermediate-tensor memory traffic.
- Applying RoPE by explicitly splitting QKV, calling separate rotary ops on Q/K/V, then re-concatenating — instead of routing the unsplit tensor through a single fused rotary kernel (fused QKV RoPE / fused MLA-YARN RoPE).
- Per-expert Python loop over MoE experts running one small GEMM each (`for expert in local_experts: expert(hidden)`), underutilizing the GPU when per-expert token counts are small, instead of a batched grouped GEMM / GroupedLinear.
- MoE token dispatch / routing done with eager index-based permute, torch.split + torch.cat over Python lists, and `.tolist()` calls that force host-device sync, instead of a single fused permute / indices-to-multihot kernel.
- Separate LayerNorm module followed by a column-parallel GEMM (norm then fc1), or activation-func casts, kept as distinct modules instead of a fused LayerNormLinear / LayerNormMLP op (especially costly under FP8 due to extra FP8->FP32 casts).

**检测时该查什么（父快照核验）**：
- Is this a hot path (per-layer / per-token: activation, RoPE, router, token dispatch, expert GEMM, bias-dropout-add)? Unfused eager ops here launch many small kernels every step.
- At parent snapshot, is the fusion opt-in and default-disabled (config flags like apply_rope_fusion, bias_activation_fusion, moe_permute_fusion, moe_grouped_gemm, use_te_*, fused_single_qkv_rope)? If the flag is off/absent, the slow unfused path is the baseline.
- Are elementwise ops sitting outside a @jit_fuser/torch.jit.script/torch.compile scope, or split across a conditional so the fuser cannot combine them (e.g. bias-add separated from dropout/add)?
- Does the code force CPU-GPU sync in a fusable region (.tolist(), Python-list split/cat, per-expert Python loop)?
- Check dependency/version gates: TransformerEngine version, apex, flash-attn, grouped_gemm/CUTLASS, torch>=2.2 for torch.compile — a missing dependency silently falls back to the unfused path.
- Is the fused kernel guarded to a specific dtype/hardware (bf16-only grouped GEMM, CUDA-only Triton kernels, FP8-specific benefit)? Outside those constraints the unfused path runs.

**高发文件 top5**：`megatron/core/transformer/moe/*.py (moe_utils.py, router.py, token_dispatcher.py, grouped_mlp / experts)`, `megatron/core/fusions/*.py (fused_bias_swiglu.py, fused_bias_geglu.py, fused_bias_dropout.py, fused_*_rope_apply.py)`, `megatron/core/transformer/mlp.py (activation-func dispatch, bias_activation_fusion wiring)`, `megatron/core/transformer/attention.py + RoPE modules (rotary/yarn embedding, KV-cache append, flash_decode path)`, `megatron/core/jit.py and TE spec wiring (@jit_fuser decorator, TELayernormMLP / TELayerNormColumnParallelLinear specs)`

### kernel-noop-skip  （29 张 train 卡）

**典型反模式**：
- Unconditionally applying an identity/no-op transform that materializes a new tensor: e.g. `repeat_interleave(ratio, dim=...)` or broadcast-expand where ratio==1 (non-GQA/MHA), copying KV heads every forward instead of gating on `ratio > 1`.
- Allocating a zero-initialized output/scratch buffer (`torch.zeros(...)`) every forward when the consumer fully overwrites it — the memset kernel is pure waste; use `torch.empty(...)` or skip when count/size is 0.
- Applying a full-tensor elementwise op with a default identity operand on the hot path: `logits * scale`, `logits.div_(temperature)`, `g.mul(1.0/factor)` without a `if scale != 1.0` / `if temperature != 1` / greedy-batch guard.
- Launching a kernel / building device tensors / running collectives unconditionally per step for inputs that are commonly empty or inactive: empty block-swap maps, batches with no active LoRA (`token_lora_mapping all == -1`), padded slots, empty seq-group batches, prefix-cache-miss allocation fan-out.
- Iterating over padded/masked ranges (grid over padded_num_slices, full K-tile loop in non-EVEN_K, all KV tiles under sliding-window, all k-tiles in causal attention) instead of computing the live range and skipping masked/padded work with an early return.

**显现条件 top3**：
- Config makes a scaling/repeat factor equal to identity (ratio==1 for non-GQA/MHA, temperature==0 all-greedy batch, scale==1.0, gradient_predivide_factor==1.0), so the op is a semantic no-op yet still launches a kernel/copy every step.
- Common-case degenerate inputs on the hot per-step path: empty batches, empty block-swap/copy maps, prefix-cache miss (empty sentinel), zero-count modality dummies, all-base-model (no active LoRA) batches, padded CUDA-graph/recompilation-bucket slots.
- Padded/masked iteration space larger than live work: sliding-window or causal attention over full tile loops, grid launched over padded_num_slices, non-EVEN_K masked loads on all blocks; benefit scales with vocab_size / sequence length / gap between actual and padded counts.

**历史量级样本**：
- 3.9% TTFT improvement (vllm DeepGEMM MoE expert_ids init)

**检测时该查什么（父快照核验）**：
- Is this op on a per-forward / per-step hot path, or per-request allocation path? (repeat, buffer alloc, elementwise scale, collective, kernel launch)
- Does the transform become an identity when a config/runtime scalar hits its default (ratio==1, scale==1.0, temperature==0/1, factor==world_size)? If so is there a guard skipping it?
- Is the output buffer fully overwritten by the consumer? If yes, is it needlessly zero-initialized (torch.zeros vs torch.empty)?
- For the common/default caller, is the input degenerate (empty batch, empty mapping, count==0, cache miss, no active adapter)? Is there an early return before device work?
- Does the kernel/grid iterate over padded or fully-masked ranges (padded_num_slices, non-EVEN_K, sliding-window/causal tiles)? Could the live range be computed to skip them?
- Does the guarded/unguarded path force a device-side cost even for the no-op case (memset, torch.cuda.synchronize, GPU->CPU sync, extra autograd node, new tensor copy)?

**高发文件 top5**：`Megatron-LM: megatron/**/attention*.py (ParallelAttention/Attention.forward, GQA repeat_interleave), transformer.py (make_viewless_tensor), tensor/sequence-parallel linear layers`, `vllm: vllm/attention/layer.py (Attention/MLAAttention output buffer), attention Triton/Pallas/Neuron kernels (unified_attention, kv_cache_update, paged attention)`, `vllm: vllm/lora/**/*.py + LoRA shrink/expand & fused_moe_lora Triton/CUDA kernels`, `vllm: vllm/v1/sampler & spec-decode/logprobs paths (Sampler.forward apply_temperature, _get_logprobs, compute_probs, LogitsProcessor.forward)`, `vllm: KVCacheManager (allocate_slots), worker/scheduler (empty-batch early return), TPUWorker.prepare_worker_input; DeepSpeed ZeRO-3 reduce/average_tensor; TransformerEngine BasicOperation.get_extra_state`

### kernel-occupancy-redesign  （68 张 train 卡）

**典型反模式**：
- Scalar element-per-thread load/store loops in memory-bound elementwise/norm/cache kernels (e.g. `for (idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) out[idx] = f(in[idx])`), issuing one narrow global transaction per element instead of coalesced 128-bit vector accesses.
- Runtime scalar parameters (scoring func, activation type, dtype/mode flags) tested per-element inside the inner loop (`if (mode == X)`), forcing branch evaluation and blocking constant folding instead of being lifted to compile-time template/constexpr specialization.
- Fixed/hardcoded launch geometry and tile sizes decoupled from the actual problem shape: static YTILE/block-size macros, one-program-per-row or one-program-per-request grids, capped num_splits, and tile shape coupled to an unrelated parameter (e.g. KV paging block size) — leaving SMs idle for tall-skinny GEMMs, small-M/small-batch, or large-vocab/few-request cases.
- Serial single-thread work inside a parallel kernel: prefix-sum/init/fill done only by `threadIdx.x==0` or a serial fill loop at the start of the same block that stalls the rest of the block; no SplitK on large-K tall-skinny matmuls so the reduction dimension is left unparallelized.
- Passing reinterpret_cast'd vector pointer elements straight into the op so the compiler emits multiple narrow loads instead of a single wide vectorized load; per-block cross-sequence masking that could be avoided by chunk-aligned data layout.

**显现条件 top3**：
- Memory-bound kernels (elementwise activation/gating, RMS/LayerNorm, KV reshape_and_cache, MLA gather/concat) on fp16/bf16/fp8 where hidden_size/head_size/element_count is divisible by the vector width (VEC_SIZE 8 for 2-byte, 4 for fp32) and pointers are 16-byte aligned.
- Low-occupancy shapes: tall-skinny GEMMs (large K, small N e.g. LoRA rank), small-M/low-batch decode (M<=64), few requests over large vocabulary, or small KV paging block sizes — too few tiles/programs launched to fill the SMs.
- MoE / multi-LoRA and expert-parallel routing paths (grouped_topk, moe_align, fused_moe_lora shrink/expand) where per-element runtime branches, serial prefix-sums, or missing SplitK dominate; often arch-gated (SM90+ PDL, SM100 grouped GEMM).

**检测时该查什么（父快照核验）**：
- Is this a memory-bound kernel doing scalar element-per-thread loads/stores? Check whether reads/writes go through a vectorized-with-alignment helper (int4/128-bit) vs one scalar per iteration.
- Are runtime mode/dtype/activation flags tested inside the inner per-element loop instead of being template/constexpr parameters dispatched once at launch?
- Is launch geometry (grid, block size, tile/YTILE, num_splits) static or derived from an unrelated parameter (e.g. KV block size), rather than chosen from the actual M/N/K/vocab/SM-count?
- Does the kernel leave a whole dimension unparallelized — serial prefix-sum/fill by a single thread, or a large-K reduction with no SplitK on a tall-skinny matmul?
- Is this a hot path (per-token decode, per-layer norm, KV cache write, MoE routing) where the underutilization multiplies across steps/layers?
- For aligned wide loads: does the vectorized helper actually compile to a single 128-bit transaction, or does casting/indexing defeat it into multiple narrow loads?

**高发文件 top5**：`csrc/**/*.cu — activation, layernorm/rms_norm, reshape_and_cache, cache-concat/gather kernels`, `csrc/quantization/** and cutlass_* — FP8 GEMM / grouped-GEMM MoE dispatch (sm90/sm100) and vectorize_with_alignment helpers`, `vllm/lora/ops/triton_ops/** — LoRA shrink/expand and fused_moe_lora GEMM (SplitK, swizzle, PDL)`, `csrc/moe/** — moe_align_block_size, grouped_topk routing, moe_lora_align kernels`, `Triton kernels for norm/attention/mamba/sampling (layernorm_guard.py, unified attention, Mamba2 SSD, gumbel/argmax sampler)`

### suboptimal-kernel-gate  （247 张 train 卡）

**典型反模式**：
- Import-fallback block imports the fast-kernel symbol from the *same* module whose failure triggered the fallback (e.g. re-importing from the FA3/Hopper module in its own except branch), so the fast path is never actually reached.
- Guarding a fused-kernel branch on an undefined or misspelled capability flag (e.g. checking `HAVE_ROPE_FUSION` when the import block only ever sets `HAVE_APPLY_ROPE_FUSION`), causing the fused path to silently never engage or to crash.
- Hardcoding a single upstream symbol name for a fast kernel that the dependency later renamed (e.g. `flash_attn_unpadded_func` -> `flash_attn_varlen_func`), so `X = None` on import failure permanently disables the fast path on newer library versions.
- Path selector logic that over-broadly routes a whole feature onto the slow path (e.g. `if not use_flash or group_query_attention:` forcing all GQA — including the degenerate num_groups==num_heads case — onto unfused attention).
- Unconditional hard-block asserts / ValueErrors that forbid a valid fast-kernel combo (fp8+grouped-GEMM MoE, flash-attn for BERT, TE for fp16, interleaved-RoPE + rope-fusion), pinning the run to a higher-precision or unfused path.
- Binding a train/inference-specialized fused function once in __init__ (self.training==True) so eval/validation never swaps to the correct variant, keeping the wrong fusion active.
- Replacing a fast GPU kernel with a semantically-equivalent-but-slower one in a hot path (e.g. torch.histc -> torch.bincount, F.embedding -> advanced indexing) without gating the swap behind a determinism/config flag.

**显现条件 top3**：
- Opt-in fusion/fast-kernel flag enabled but gate silently falls back to slow path (broken import or misspelled flag).
- Newer/alternate dependency version or package name breaks a hardcoded import, disabling the fast kernel.
- Hardware/dtype/shape combo not covered by dispatch table falls off the fast path.

**检测时该查什么（父快照核验）**：
- At parent snapshot, check every `try/except` kernel-import block: does the except branch import the SAME symbol/module that just failed? Does it set the flag name the guard later reads (spelling matches)?
- Trace each capability flag (HAVE_*, use_flash, fp8) from where it is set to where it is checked — confirm the guarded fast-kernel branch is actually reachable when the feature is enabled.
- Is this an inference/training hot path (attention forward, RoPE apply, wgrad GEMM, MoE token dispatch, softmax, embedding)? A wrong gate here costs per-forward/per-step throughput.
- Check dispatch tables and dtype/shape switch statements for missing cases (bf16, longer seq_len, fp8 alignment) that silently raise or fall to a slow else-branch.
- Check for hard asserts / ValueErrors that block valid fast-kernel combos, and whether removing them requires plumbing (padding/unpadding, alignment) to stay on the fast path.
- Check whether train/eval-specialized fused functions are re-selected on model.eval() rather than bound once in __init__.
- Is the fast path default-enabled or opt-in? Opt-in gates that silently no-op are low-detectability regressions (no error, just slow).

**高发文件 top5**：`megatron/core/transformer/attention.py`, `megatron/core/transformer/custom_layers/transformer_engine.py (TE import/wrappers)`, `megatron/core/models/**/rotary_pos_embedding / rope import blocks`, `megatron/core/tensor_parallel/layers.py (LinearWithGradAccumulation, VocabParallelEmbedding)`, `megatron/core/transformer/moe/** (router / token_dispatcher) and fused softmax CUDA dispatch (scaled_masked_softmax*)`

### unfused-transpose-copy  （42 张 train 卡）

**典型反模式**：
- Forcing a `.contiguous()` (or `.transpose(...).contiguous()`) on q/k/v or hidden-state tensors every forward pass to satisfy a downstream kernel's layout requirement (e.g. bshd/2D-row-major), materializing a full extra copy on the hot path.
- Chaining reshape/permute/view/transpose ops to reorder tokens or heads (e.g. permute(2,0,1).view(...).permute(1,0).unsqueeze(1)) and dispatching via gather/scatter/index_copy instead of using a fused/custom kernel that consumes the native layout.
- Concatenating a broadcast/expanded (non-contiguous) tensor via torch.cat, forcing materialization, instead of pre-allocating the output and copying slices in place.
- Explicitly transposing weight/operand layouts before a GEMM/grouped-GEMM call (trans_b=True or w.transpose(1,2)) when the kernel could accept the native layout or a strided view, forcing a non-fused/slow path.
- A Python wrapper unconditionally calling input.contiguous() before a CUDA kernel because the kernel only accepts 2D row-major / contiguous input, rather than passing stride params so the kernel handles strided/sliced tensors directly.

**显现条件 top3**：
- Attention / RoPE paths where q/k/v are strided or sliced views (MLA q_pe/k_pe splits, fused-RoPE bshd layout, packed THD/CP paths) and a kernel requires a specific contiguous layout — a per-forward transpose+contiguous is inserted.
- MoE token permute/unpermute and grouped-GEMM paths (expert/tensor/sequence parallel) where mask/index gather-scatter and explicit weight transposes reorder tokens or operands before the expert kernel.
- torch.compile / Inductor or kernel-layout constraints (e.g. requires_contiguous tag, row-major-only kernels) forcing column-major quantized weights or strided inputs through an extra .contiguous() materialization.

**检测时该查什么（父快照核验）**：
- Is a `.contiguous()` / `.transpose().contiguous()` sitting on the per-forward hot path (attention forward, per-pipeline-stage, per-GEMM), i.e. re-executed every step rather than once at init?
- Is the copy a workaround (WAR) for a kernel that only accepts one layout? Check whether the kernel/wrapper could instead take a stride/head_stride parameter or a transpose_output_memory flag to consume the native layout.
- Are transpose/permute/reshape chains feeding gather/scatter/index_copy in MoE permute/unpermute, or trans_b/weight-transpose feeding a GEMM — could a fused custom autograd op or native-layout kernel avoid the reorder copy?
- Is the transpose default-enabled (feature flag on by default, e.g. apply_rope_fusion=True) so the extra copy affects the common configuration?
- Does the copy scale with a parallelism dimension (pipeline stages, world_size, num_experts) so the aggregate cost grows with the deployment?
- Is a broadcast/expanded tensor being torch.cat'd (forcing materialization) where a pre-allocated buffer + in-place slice copy would avoid it?

**高发文件 top5**：`Megatron-LM: */transformer/custom_layers/*attention* / TEDotProductAttention forward (fused-RoPE bshd transpose)`, `Megatron-LM: */transformer/moe/moe_utils.py, base_moe_layer.py, grouped MLP (permute/unpermute, grouped GEMM trans_b)`, `Megatron-LM: */transformer/transformer.py (ParallelTransformer.forward per-stage [b,s,h]<->[s,b,h] transpose) and mlp.py (SwiGLU chunk)`, `vllm: vllm/_custom_ops.py and RoPE/rotary_embedding CUDA kernels (head_stride / strided q/k support)`, `vllm: attention/quantization/MoE kernel wrappers (rms_norm wrapper, MLA prefill k concat, LoRA set_lora copy, TPU Pallas gmm, quantized GEMM contiguous)`


## 大类：memory-footprint

### activation-recompute  （24 张 train 卡）

**典型反模式**：
- Wrapping a module's forward in a checkpoint/recompute primitive (mpu.checkpoint / tensor_parallel.checkpoint / te_checkpoint / fleet.utils.recompute) without checking whether inputs require grad — layers whose hidden_states.requires_grad is False still get re-entrant-checkpointed, paying full recompute FLOPs for zero memory benefit.
- Double-recomputing kernels that already recompute internally in their own backward (e.g. wrapping FlashAttention in an activation-checkpoint forward) — redundant forward FLOPs with no extra memory saved.
- Config default mismatch between argparse default and TransformerConfig field default for recompute knobs (e.g. --recompute-num-layers argparse=1 vs field=None), silently changing uniform-recompute window size and thus recompute FLOPs.
- Deprecated/legacy flags that silently map onto full activation recomputation (whole-block recompute) instead of selective recompute, quietly trading throughput for memory.
- Hardcoded is-checkpointable predicate that short-circuits on model class name and ignores user-supplied checkpointable_layers, so extra layer types silently stop being checkpointed and activation memory grows.
- FP8/quantized recompute paths that fail to stash/restore amax_history/scale/scale_inv across the recomputed forward, or custom GEMM primitives that standard remat policies (dots_with_no_batch_dims) cannot match, so intended activations are not saved/rematerialized.
- Output-retaining recompute primitives that keep intermediate tensor storage alive instead of freeing it (missing untyped_storage().resize_(0) or discard-output pattern), defeating the memory saving.

**显现条件 top3**：
- FP8 / quantized-recipe training (DelayedScaling/CurrentScaling/MXFP8/blockwise), where recompute must also carry FP8 meta (amax/scale) and where redundant recompute wrappers or missing meta-stashing surface as memory or throughput regressions.
- Tensor/pipeline/context/expert parallelism enabled (TP>1, PP>1, CP>1, MoE/EP): recompute window sizing, distributed activation sharding, and per-stage requires_grad interactions govern whether the memory-for-compute trade actually pays off.
- Full/uniform activation recompute enabled via config default or deprecated flag (recompute_granularity=full, method=uniform, recompute_num_layers), silently recomputing entire transformer blocks and adding forward FLOPs.

**检测时该查什么（父快照核验）**：
- Is this a hot path (per-layer / per-microbatch transformer forward)? Recompute wrappers here re-run forward in backward — verify the extra FLOPs buy real memory.
- At parent snapshot, is the recompute opt-in or default-enabled? Check argparse default vs Config field default for recompute-num-layers / granularity / method mismatches.
- Does the caller guard on hidden_states.requires_grad before wrapping in checkpoint? Non-grad inputs should skip recompute (recompute_skip_num_layers pattern).
- Does the kernel/module already recompute internally (FlashAttention)? Avoid double recompute wrappers.
- For FP8/quantized paths: is FP8 meta (amax_history/scale/scale_inv) stashed and restored across the recomputed forward? Are custom GEMM primitives covered by the remat/checkpoint policy?
- Is the is-checkpointable predicate honoring user-supplied checkpointable_layers, or short-circuiting on model class name?
- Does the recompute primitive actually free output storage (resize_(0)) rather than retaining it?
- For distributed activation checkpointing: is TP size >1 asserted, and are saved activations sharded (split_tensor_into_1d_equal_chunks) and re-gathered correctly?

**高发文件 top5**：`Megatron-LM: megatron/**/transformer_block.py (_checkpointed_forward, recompute paths)`, `Megatron-LM: megatron/**/transformer/moe/moe_layer.py (moe_layer_recompute, custom_forward)`, `Megatron-LM: megatron/**/tensor_parallel/random.py (CheckpointFunction, distribute_checkpointed_activations, split_tensor_into_1d_equal_chunks)`, `Megatron-LM: megatron/**/arguments.py (--recompute-*, --checkpoint-activations, --distribute-checkpointed-activations defaults)`, `TransformerEngine: transformer_engine/{pytorch,jax,paddle}/**/{layernorm_mlp,attention}.py (checkpoint flag, checkpoint_name, FP8RecomputeBuffer); DeepSpeed: deepspeed/runtime/pipe/module.py (_is_checkpointable) & activation_checkpointing/checkpointing.py (non_reentrant_checkpoint)`

### low-precision-state  （56 张 train 卡）

**典型反模式**：
- Building sub-modules (e.g. MTP layers, VLM projector, extra transformer blocks) via `build_module`/init without entering the fp8 init context (`get_fp8_context(config, is_init=True)`) that the main block uses, so those weights silently stay in high precision and lose the fp8 memory savings.
- Unconditionally upcasting large logits/activations to fp32 before an op (e.g. `output.contiguous().float()` before `vocab_parallel_cross_entropy`, or `.to(fp32)` on saved-for-backward activations), materializing a [tokens, vocab] or [b, s, vocab] fp32 tensor when a low-precision path would suffice.
- Optimizer state/master-weight storage hardcoded to fp32 (exp_avg/exp_avg_sq/master params) instead of routing through a precision-aware optimizer that can hold bf16/fp16/fp8 moments or store only 16-bit param remainders.
- Resolving `kv_cache_dtype == "auto"` by unconditionally returning `model_config.dtype` (fp16/bf16) instead of inspecting the checkpoint's quantization_config to pick fp8 KV cache — leaving KV cache at 2x the memory it could use.
- Model/config registry keyed on a stale architecture string (e.g. mapping fp8 KV-cache config to an old `...ForCausalLM` name), so a renamed/new arch misses the low-precision lookup and falls back to the high-precision default.
- Hard `raise NotImplementedError`/capability guards that block an fp8 KV-cache or fp8-attention path for a backend/hardware combo that could otherwise support it, forcing high-precision fallback.

**显现条件 top3**：
- fp8/low-precision path exists in code but a sub-module init, backend, or default-resolution branch does not opt into it, so the tensor stays in bf16/fp16/fp32 — extra memory footprint with no functional error.
- KV cache or optimizer-state dtype left at 'auto'/fp32 default (or missed by a stale config-map key) when the checkpoint/config actually supports fp8/bf16 low-precision storage.
- Requires precision-aware/distributed optimizer or FP8-capable hardware + TransformerEngine/TRTLLM/FlashInfer version; when the version/hardware/backend gate is not met, the code silently keeps the high-precision fallback.

**检测时该查什么（父快照核验）**：
- Does every sub-module (MTP, vision projector, extra blocks) enter the same fp8 init/forward context (`get_fp8_context(..., is_init=True)`) as the main transformer block, or is one built via plain `build_module`?
- For 'auto' dtype resolution (KV cache, optimizer state), does the code inspect quantization_config/precision-aware flags to pick low precision, or does it unconditionally return `model_config.dtype`/fp32?
- Are optimizer master weights, gradients, and Adam moments (exp_avg/exp_avg_sq) routed through the precision-aware optimizer (bf16/fp16/fp8 or 16-bit remainder), or hardcoded fp32?
- Is there a hot-path `.float()`/upcast on a large activation/logits tensor ([tokens, vocab], [b, s, vocab]) that a low-precision cross-entropy or activation-store path could avoid?
- Do config-map / registry lookups key on the exact current architecture string (guard against renamed `...ForCausalLM` -> `...V32ForCausalLM` misses)?
- Are there NotImplementedError/capability guards disabling fp8 KV-cache or fp8-attention for a backend (FlashInfer/Triton/ROCm AITER/FlashMLA) or SM level that actually supports it?
- Is the low-precision feature default-off (opt-in flag/env var) or default-on, and is the required TE/CUDA/backend version present?

**高发文件 top5**：`Megatron-LM: megatron/core/models/**/multi_token_prediction.py, multimodal projector (fp8 init context wrapping)`, `Megatron-LM: megatron/core/optimizer/*.py (precision-aware optimizer / FusedAdam store_param_remainders, bf16 exp_avg states)`, `Megatron-LM: megatron/core/fusions/*swiglu*, model cross-entropy / lm-head logits path (fp16_lm_cross_entropy)`, `vllm: vllm/**/attention/backends/*.py (flashinfer.py, flashmla, mla, triton, rocm aiter) fp8 KV-cache / fp8-attention wiring`, `vllm: vllm/config / arg_utils.py kv_cache_dtype resolution helpers (kv_cache_dtype_str_to_dtype, resolve_kv_cache_dtype_string, MODELS_CONFIG_MAP)`

### meta-device-init  （5 张 train 卡）

**典型反模式**：
- Instantiating a module with real tensors on CPU/default device and then moving to GPU (e.g. `model = ModelClass(...)` followed by `model.cuda()`), which transiently allocates the full weight set in host memory before the device copy.
- Constructing throwaway sub-modules with real allocations that are immediately overwritten (e.g. `dummy = torch.nn.Linear(in, out); dummy.weight = param`) instead of building them under a meta/no-alloc context.
- Unconditionally materializing meta-device parameters into real GPU allocations (e.g. a `to_empty_if_meta_device(..., device=cuda)` guard) even on parallelism paths (FSDP/sharded init) that already materialize and shard parameters themselves — causing double allocation.
- Every rank fully materializing duplicate full (unsharded) parameters at init time before sharding, rather than deferring materialization via `device='meta'` + a `reset_parameters()` hook invoked after sharding.
- Defaulting weight-init to a path that builds the full master weight redundantly (e.g. CPU full master-weight construction cast to params_dtype) without an opt-in to the leaner device-init strategy.

**显现条件 top3**：
- Sharded/distributed init paths (FSDP, Megatron FSDP, tensor-parallel) where parameters are already materialized+sharded downstream, so eager meta-materialization or full-tensor init double-allocates.
- Model-loading / weight-quantization / construction phase on CUDA, where transient CPU allocation or default-device allocation precedes the intended device placement.
- meta-device deferred-init enabled but its interaction with a materialization guard or init strategy is misconfigured (e.g. init_model_with_meta_device with an FSDP path).

**历史量级样本**：
- Pre-FSDP memory: 83.9MiB → 0.0MiB (README example on 8xL40S, TransformerEngine deferred meta init)

**检测时该查什么（父快照核验）**：
- Is module construction wrapped in a meta/no-alloc context (`torch.device('meta')` or `torch.device('cuda')`) or does it allocate real tensors on the default/CPU device first?
- Does any post-construction guard (e.g. to_empty_if_meta_device / .cuda()) unconditionally materialize params on a path (FSDP/sharded) that will materialize/shard them itself?
- Are throwaway/dummy modules created purely to borrow shape, and do they allocate real weights that are immediately overwritten?
- Is the default init strategy building a full master weight (e.g. CPU fp32 then cast) per-rank rather than deferring to sharded materialization?
- Is deferred init (device='meta' + reset_parameters()) actually reached, or bypassed by an earlier eager-materialization branch?
- Does behavior differ across TP world sizes / ranks such that each rank redundantly materializes the full unsharded parameter set?

**高发文件 top5**：`megatron/**/arguments.py (init-strategy defaults / opt-in flags)`, `megatron/**/*get_model*, training/model setup paths (meta-materialization guards)`, `vllm/model_executor/**/loader*.py (model construction / device placement)`, `vllm/model_executor/layers/quantization/**/torchao*.py (dummy-module weight quant)`, `transformer_engine/**/module*.py (device='meta' deferred init + reset_parameters)`

### offload-management  （43 张 train 卡）

**典型反模式**：
- Enabling an offload feature by default (e.g. defaulting `cpu_offloading=True` / non-zero offload layers, or offload flags defaulting True in the config dataclass), silently incurring D2H/H2D copies on every layer's activations/weights even for workloads that fit in device memory.
- Offloading synchronously without double-buffering / async prefetch: hardcoding the double-buffer switch off or omitting a second pinned CPU buffer, so the offload/reload of tensors serializes against GPU compute instead of overlapping it, creating bubbles.
- Moving state to device in an exchange/gather path (`.cuda()`) but never restoring it back to CPU when the input originally lived on host, so all gathered/loaded shards permanently stay GPU-resident and leak device memory.
- Re-entering a memory-pool / wake-up context on a region that is already restored/awake (e.g. wrapping reload in `use_memory_pool(tag=...)` after a partial `wake_up(tags=...)`), double-mapping the same region and corrupting/inflating the memory accounting.
- Allocating optimizer/momentum buffers on CPU but recreating a fresh GPU copy plus copy-back every step (`narrow().to(device)...copy_(buffer)`), turning offload into per-step host<->device churn instead of a one-time transfer.

**显现条件 top3**：
- Offload feature enabled (cpu_offloading / cpu_offload_gb>0 / sleep-mode wake_up / ZeRO CPU optimizer offload) on GPU with host<->device PCIe transfers — throughput/bubble regression when transfers don't overlap compute.
- Tensor-parallel or distributed-checkpoint paths that gather/broadcast shards to GPU (`.cuda()` in exchange) or keep TP weights on host — device-memory leak or unexpected residency when input originated on CPU.
- Large-model / memory-bound configs where offload is the enabling capability (very large checkpoints, big dataset index arrays); note PP>1 and activation-recompute are frequently incompatible with activation offloading.

**历史量级样本**：
- iteration time 1500ms --> 910ms (finetune Qwen2.5-3B on 2xA100, Muon momentum buffer on-device fix)
- ~500 TFLOPS on a single GH200 (SuperOffload full fine-tune of GPT-OSS-20B/Qwen3-14B/Phi-4) — absolute capability figure, not before/after delta

**检测时该查什么（父快照核验）**：
- Is the offload feature default-enabled? Check the config dataclass / CLI validate_args (flags defaulting True, non-zero offload-layer counts) — a default flip here regresses every model.
- Is the offload/reload on the training hot path (per-layer forward, per-optimizer-step)? Verify double-buffering / async prefetch (second pinned buffer, prefetch_commit_async, non_blocking copies) is actually wired, not hardcoded off.
- In gather/exchange/wake-up paths, is device residency symmetric? Confirm tensors moved to GPU are moved back to host when input was CPU-resident, and that already-awake regions aren't re-entered into a memory pool.
- Are compatibility guards checked (PP>1 forbidden, activation recompute incompatible, TE/UVM/pinned-memory availability, engine version like TE>=2.5.0) before the offload path is taken?
- For host-RAM footprint: are large index/state arrays memory-mapped (mmap_mode) vs fully np.load'ed, especially with multiple ranks per node?

**高发文件 top5**：`megatron/core/transformer/transformer_block.py (offload_context / get_cpu_offload_context wiring)`, `megatron/core/model_parallel_config.py & transformer_config.py (cpu_offloading* flags, defaults, __post_init__ guards)`, `megatron/core/dist_checkpointing/strategies/fully_parallel.py (FullyParallelLoadStrategyWrapper exchange / .cuda residency)`, `vllm memory allocator / worker (CuMemAllocator.wake_up, use_memory_pool tags, cpu_offload UVA view)`, `deepspeed/runtime/zero + offload optimizer workers (ZeRO-Offload, ZenFlow/SuperOffload CPUAdam worker, pinned grad buffers)`

### skip-saved-tensor-backward  （18 张 train 卡）

**典型反模式**：
- Forward unconditionally calls `ctx.save_for_backward(activation)` (input, ln_out, gelu_out, weight-transpose, or a high-precision cast copy) without checking whether the corresponding grad (dgrad/wgrad) will actually be needed — i.e. no `weight.requires_grad`/`inp.requires_grad` gating threaded into the autograd Function.
- Gating the training/grad path on `module.training` (or a construction-time `config.training` flag) instead of `torch.is_grad_enabled()`, so a training-mode module run under `torch.no_grad()` still takes the full save-for-backward path and retains activations.
- Materializing an extra copy of an activation inside forward — e.g. an up-cast `input.to(fp32)` for a high-precision GEMM, or an eagerly-computed columnwise/transposed FP8 copy — and letting autograd save that fatter tensor for backward.
- Creating a large base/output buffer with `requires_grad=True` (then feeding it to `scatter_add`/in-place ops) so it becomes a grad-requiring leaf autograd retains, or holding activation tensors alive in a list across the whole warmup+steady 1F1B window instead of freeing `.data` after send.

**显现条件 top3**：
- Inference / eval or frozen-weight / frozen-input paths (requires_grad=False, or training-mode module executed under torch.no_grad()) where the saved backward tensor is never consumed but is still retained.
- FP8 / quantized linear layers (columnwise/transpose copies, weight-transpose caching) and mixed-precision routers where an extra high-precision or transposed copy of the activation is saved for the wgrad GEMM.
- Tensor-/sequence-parallel and pipeline-parallel setups where saved activations (or output tensors held across the schedule) dominate peak memory; savings scale with activation size (seq_length × micro_batch × hidden).

**历史量级样本**：
- Megatron MoE (moe_router_dtype=fp32): mem-max-allocated-bytes 3328506368→3281624576 (~1.4%) in gpt3_moe ep8; 1154966528→1142194688 in tp4_ep2
- DeepSpeed DeepCompile ZeRO3 (8x RTX 5090): eager+autocast peak 12.07→9.96 GB; eager+bf16 9.47→7.30 GB (time 551.92→571.24 ms and 419.87→445.76 ms)

**检测时该查什么（父快照核验）**：
- At the parent snapshot, check whether the forward's `ctx.save_for_backward` / saved-tensor set is gated on the actual need for that grad (`weight.requires_grad`, `inp.requires_grad`, `first_op_requiring_backward`) or saved unconditionally.
- Check whether the train/grad dispatch uses `torch.is_grad_enabled()` vs a static `module.training` / `config.training` flag — the latter leaks the full save path into no_grad inference.
- Look for extra activation copies created inside forward (up-cast to fp32/fp64, columnwise/transpose FP8, eager weight transpose) that are then saved for backward when a cheaper or no save would suffice.
- Check for buffers allocated with requires_grad=True that feed in-place/scatter ops, and for activation tensors held in schedule-level lists (1F1B output_tensors) rather than freed after their downstream send.
- Confirm this is on a hot path (per-layer linear/MLP forward, MoE router/permute, per-microbatch pipeline stage) so the retained tensor multiplies across layers/microbatches into peak memory.

**高发文件 top5**：`TransformerEngine: transformer_engine/pytorch/module/{linear,layernorm_linear,layernorm_mlp}.py (_Linear/_LayerNormLinear/_LayerNormMLP autograd Functions)`, `TransformerEngine: transformer_engine/pytorch/ops/**/basic_linear.py and the ops fusion infra (OperationFuser / first_op_requiring_backward)`, `Megatron-LM: megatron/core/tensor_parallel/layers.py (ColumnParallelLinear/RowParallelLinear, LinearWithFrozenWeight) and megatron/core/transformer/moe/**/moe_utils.py (permute/unpermute, router)`, `Megatron-LM: megatron/core/pipeline_parallel/schedules.py (1F1B output_tensors / free_output_tensor) and dist-checkpointing fully-parallel load strategy`, `DeepSpeed: deepspeed/ops/transformer/**/transformer.py (DeepSpeedTransformerLayer/Function training flag) and DeepCompile ZeRO3 partitioner / TiledFusedLogitsLoss`


## 大类：memory-management

### allocator-config  （7 张 train 卡）

**典型反模式**：
- Removing an eager pre-allocation of optimizer/state tensors (e.g. exp_avg / exp_avg_sq created via zeros_like at init) that had existed specifically to reduce allocator fragmentation; lazy first-touch allocation then re-fragments the caching allocator.
- Calling torch.cuda.empty_cache() every training/eval iteration as a fragmentation band-aid: it forces a device sync and drops the caching allocator's reserved blocks, adding a per-iter cost and re-allocation churn.
- Flipping an allocator env default (e.g. NCCL_CUMEM_ENABLE) unconditionally without honoring a user-set value, trading memory for throughput (or vice versa) globally across all hardware/NCCL versions.
- Per-object heap malloc/free on a hot creation/destruction path (new/delete per tensor wrapper in a loop) instead of a pooled/pre-reserved allocator.

**显现条件 top3**：
- Multi-GPU NCCL collectives (allreduce/allgather/reducescatter) with cudagraph enabled, on CUDA GPUs where the allocator-env default determines the memory/throughput tradeoff (cuMem enable/disable).
- Long-running training/eval loops on CUDA where fragmentation of the caching allocator accumulates over iterations (or where per-iter empty_cache is toggled on).
- High-frequency CPU/host-side tensor-wrapper create/destroy paths where per-object malloc/free dominates latency.

**历史量级样本**：
- ~1-2 GiB GPU memory cost attributed to enabling the NCCL cuMem allocator with cudagraph+allreduce

**检测时该查什么（父快照核验）**：
- Is an eager state/buffer pre-allocation being removed? Check whether it was originally added to control allocator fragmentation before deleting.
- Does the change flip an allocator-related env var default (NCCL_CUMEM_ENABLE and similar)? Confirm it is opt-in / honors user-set values and note it trades ~1-2 GiB GPU memory vs collective throughput.
- Is empty_cache()/cache-drop called inside a per-iteration hot loop (train_step / evaluate)? It forces a sync and defeats the caching allocator.
- Is object allocation on a hot path using per-object new/delete instead of a pool? Check creation/destruction loops (pack create/destroy).
- Is the new default enabled by default or opt-in, and is it hardware/NCCL-version-gated (multi-node NVLink special-casing)?

**高发文件 top5**：`Megatron-LM: optimizer construction (get_megatron_optimizer / optimizer/*.py)`, `Megatron-LM: training/eval loop (training.py train_step & evaluate)`, `vllm: env_override.py (NCCL env defaults)`, `vllm: worker/*.py (WorkerWrapperBase.init_worker)`, `TransformerEngine: C++ tensor/NVTETensor allocation core (tensor pack create/destroy, TensorAllocator)`

### buffer-sizing-layout  （80 张 train 卡）

**典型反模式**：
- Allocating buffers/tensors sized by a *coarse upper bound* rather than the actual runtime extent — e.g. padding sequences to the full model max length instead of (max_prompt + max_gen), or sizing per-request KV/encoder allocations by a batch-total budget instead of per-item token count.
- Packing params/grads/activations into a contiguous buffer with per-element or per-bucket *unaligned* start offsets, so the buffer is not a multiple of the required alignment (e.g. 16/32/64/128/256-byte or dp_world_size divisibility) — forcing cuBLAS/TE/TMA kernels onto a slower/incompatible path or breaking sharding asserts.
- Leaving projected/split QKV or attention operands as non-contiguous strided views (torch.split / last-dim scalar index / transpose) and feeding them directly to fused/flash attention or GEMM, triggering hidden re-layout copies or fallback kernels.
- Using an oversized dense fixed-dtype tensor for a naturally sparse/small-range quantity — e.g. [max_num_reqs, vocab_size] int32 bin-counts, or unconditionally int64 index arrays when values fit in int32/bitpacked form.
- Zero-copy views (frombuffer / non_blocking .to('cpu')) over a large parent tensor that pin or retain the entire backing allocation, so a small slice keeps a big buffer alive.

**显现条件 top3**：
- Low-precision GEMM (FP8/mxfp8/NVFP4) on MoE or TP layers with per-expert/per-partition dims not multiple of alignment.
- Distributed-optimizer contiguous param/grad buffers with unaligned offsets or non-divisible buckets under data parallelism.
- Inference/prefill KV-cache/encoder buffers sized by a max or batch-total budget rather than actual per-request extent.

**检测时该查什么（父快照核验）**：
- Is the buffer size derived from a runtime-actual extent or from a static upper bound / batch-total budget? Look for seq_length, max_num_batched_tokens, encoder_budget used to size per-request allocations.
- For any FP8/FP4/MoE/TP GEMM operand: are per-expert token counts and per-partition dims padded to the kernel's required alignment (16/32/64), and does the alignment constant match the dtype/hardware (TMA 16-byte, FP4=64)?
- In contiguous param/grad buffers: are per-param start offsets and per-bucket sizes both aligned (128/256-byte) AND divisible by data_parallel_world_size before shard_buffer()?
- For QKV/attention operands passed to fused/flash kernels: are query/key/value contiguous, or are they strided views from split/transpose that force a copy or fallback path?
- For large index/count tensors: could the dtype be narrowed (int64->int32) or the layout bitpacked/sparsified given the value range and shape?
- Do any zero-copy views (.to('cpu', non_blocking), frombuffer) over slices keep a large parent allocation alive or race with buffer reuse?
- Is the change opt-in (default off) or on a hot path (default enabled)? Alignment/chunk knobs here are frequently gated behind new flags.

**高发文件 top5**：`megatron/core/distributed/param_and_grad_buffer.py (ParamAndGradBuffer / grad buffer padding & bucketing)`, `megatron/core/transformer/**/attention.py, transformer_engine.py (QKV split/contiguity, TEDotProductAttention)`, `megatron/core/transformer/moe/** (MoE dispatcher, GroupedMLP, router padding for fp8/fp4)`, `vllm/v1/core/**/kv_cache_*.py (KV cache group sizing, uniform/hybrid/cross-layer layout allocation)`, `vllm/model_executor/layers/fused_moe/** and quantization modelopt (FlashInfer MoE intermediate padding)`

### deferred-free-peak  （43 张 train 卡）

**典型反模式**：
- Holding a live Python reference to a large activation/intermediate tensor for longer than needed (e.g. binding a projection output to a new local while the input stays referenced, or keeping the decoder input alive across the whole decoder stack), so two large buffers coexist at peak.
- Pseudo-freeing a tensor's `.data` for pipeline memory optimization without asserting it is viewless (`._base is None`): freeing a view (from persistent LayerNorm, scatter_to_sequence_parallel_region, or a zero-layer passthrough stage) aliases the original buffer and cannot actually reclaim it.
- Eagerly allocating destination/placeholder tensors (`torch.empty` on 'cuda', empty shards for ranks that don't own data) up front before an exchange/broadcast loop, and not `del`-ing them promptly after the async op is queued.
- Retaining factory objects or shallow-structure copies that still hold `.data` tensors until end-of-function (e.g. `sh_ten_factories` kept until `apply_factory_merges`, or `extract_matching_values` copies), pinning tensors that could be unlinked earlier.
- Calling `zero_grad()` without `set_to_none=True`, leaving fp32 main-grad copies allocated across the zero_grad boundary; or copying model grads into main grads without setting `model_param.grad = None` afterward.
- Building a full high-precision weight buffer and quantizing to low precision only after all chunks are loaded, so peak holds the entire high-precision weight; also applies to processing large batches (video frames) in one shot instead of micro-batches.

**显现条件 top3**：
- Pipeline model parallelism (PP>1), especially 1F1B / interleaved schedules where output activations are held until backward and the deallocate/free-output optimization is gated off or defeated by views.
- Weight loading / checkpoint save/load paths: fp8 online quantization (high-precision buffer held), dist-checkpoint exchange (eager placeholder allocation), and lazy-tensor (XLA/TPU) narrow-copy materialization.
- Optimizer/gradient boundary in mixed precision (fp16/bf16 with fp32 main grads): redundant grad copies persist across zero_grad or grad-copy steps.

**历史量级样本**：
- ~0.5% runtime cost (sequence-parallel embedding scatter viewless fix)

**检测时该查什么（父快照核验）**：
- Is this on a hot path (per-microbatch forward/backward, per-layer activation) where an extra live reference doubles peak activation memory?
- Does a memory-freeing optimization (pseudo-free of `.data`) require the tensor to be viewless? Check whether the producer returns a view (persistent LayerNorm, scatter/split ops, zero-layer passthrough) and whether `._base is None` is asserted.
- Is the deferred-free behavior gated behind a flag whose default disables the optimization (e.g. deallocate_pipeline_outputs default False)?
- At the parent snapshot, are large buffers (high-precision weights, placeholder shards, factory `.data`, fp32 main grads) held longer than necessary — could an explicit `del`, `param.grad=None`/`set_to_none=True`, `gc.collect()`, or earlier unlink reclaim them?
- Does the caller free memory promptly after an async/exchange op, or does it keep placeholders/copies alive through the whole loop?
- Is quantization/feature extraction done on the full buffer at once instead of chunked/micro-batched, inflating transient peak?

**高发文件 top5**：`megatron/core/pipeline_parallel/schedules.py (deallocate/free_output_tensor, 1F1B activation lifetime)`, `megatron/core/dist_checkpointing/**/*.py (fully-parallel load exchange, factory merges, tensorstore load)`, `megatron/core/optimizer/*.py (Float16Optimizer zero_grad / copy_model_grads_to_main_grads)`, `megatron/core/models/**/*.py & megatron/core/utils.py (GPT/Mamba forward, WrappedTensor, viewless tensor helpers)`, `vllm/model_executor/**/ (weight loading / process_weights_after_loading; layer forward activation reuse) & cumem allocator paths`

### memory-leak-unpruned-state  （77 张 train 卡）

**典型反模式**：
- Per-request state stored in module/connector-level dicts, sets, or deques (keyed by request id) that are populated on request start/enqueue but only pruned on one specific completion path — so aborts, errors, or TP-mismatch cases leak entries indefinitely.
- Applying @lru_cache (or similar memoization) to an instance/bound method: the cache keys on `self`, so it retains references to up to maxsize object instances and all their registered buffers/tensors, defeating GC.
- Recomputing/repacking a parameter into a new tensor (dequant, transpose, unpack) during process_weights_after_loading while leaving the original nn.Parameter still registered on the module, keeping dead storage resident.
- Accumulator container (state_dict={}, activation input/output lists) hoisted out of a per-file/per-microbatch loop or appended-to unconditionally (even on forward_only/eval), so it is never cleared between iterations and grows without bound.
- Storing a tensor that is a view (._base is not None) into long-lived activation/pipeline buffers, or reassigning tensor.data on a view, which keeps the entire base storage alive across iterations.
- gc.freeze() around a hot region (CUDA graph capture) without a matching collection on unfreeze, leaving unreachable objects/allocations tracked and never returned to the allocator.
- Delayed-free protocols (P/D disaggregation: free KV blocks only after remote READ/notification) with no timeout or abort fallback, so blocks pin forever if the consumer never reads.

**检测时该查什么（父快照核验）**：
- Is per-request/per-object state ever added to a dict/set/deque? Trace ALL removal paths — is there exactly one prune site that misses abort/error/cancel/TP-mismatch branches?
- Is a container (dict, list) declared at function scope but populated inside a per-file/per-shard/per-microbatch loop without being cleared between iterations?
- Is @lru_cache / memoization applied to a bound method or something that keys on a heavy object holding tensors/buffers?
- After repacking/dequantizing a weight, is the original nn.Parameter still registered (del old param / replace, or does storage stay alive)?
- Are tensors stored into long-lived buffers views (check ._base is not None) or is .data reassigned on a view?
- Is this on the hot path / default-enabled? Check env-var gates (default on vs off) and whether it runs in forward_only/eval.
- Any gc.freeze() without a compensating gc.collect() before/after unfreeze?
- Delayed-free/READ-notification block release — is there a timeout or abort fallback if the notification never arrives?

**高发文件 top5**：`vllm/distributed/kv_transfer/kv_connector/** (NIXL, LMCache, connector lifecycle/metadata binding)`, `vllm/v1/core/**scheduler** (cached request data deques, block free/skip logic, MambaManager/KV cache managers)`, `megatron/core/pipeline_parallel/schedules.py (input_tensors/output_tensors activation lists)`, `megatron/core/**/transformer/**rotary/attention** (lru_cache'd embeddings, inference KV buffers, viewless-tensor make_viewless_tensor)`, `model weight loading / quantization process_weights_after_loading (compressed-tensors MoE, torchao safetensors, FP8 FSDP2 transpose cache)`

### redundant-buffer-copy  （93 张 train 卡）

**典型反模式**：
- Out-of-place elementwise ops on large tensors where an in-place variant is safe (e.g. `x = x - m`, `x = a * b`, `x + bias` in eval/no-grad paths) — allocates a full-size intermediate instead of `x.sub_(m)`/`a.mul_(b)`/`x.add_(bias)`.
- Unconditional `.contiguous()` / `.clone()` / `.float()` / `.astype()` on hot-path tensors regardless of whether the consumer actually requires the copy (e.g. forcing q/k/v contiguous right before a kernel that accepts strided input, dtype-widening a zero-copy `frombuffer` view every sample).
- Allocating a tensor on CPU then moving to device (`torch.tensor(...).cuda()`, `torch.zeros(shape).type(...).cuda()`, `torch.empty(..., dtype=...).cuda()`) — pays a host allocation plus a host→device memcpy (and often a host/device sync) each call instead of `device=current_device()`.
- Kernel/op writes to a scratch tensor then `out.copy_(tmp)` back into the destination, or defunctionalized ops left in `auto_functionalized` wrappers that forbid in-place mutation — forcing an extra output buffer + copy-back.
- List-of-views collectives / `torch.split` / `torch.concat(list(t))` / `.repeat(...)` that materialize per-chunk copies or trigger a backward `cat` allocation, where a flat-buffer collective, view/reshape, or `expand` would avoid the copy.

**显现条件 top3**：
- Hot per-step / per-layer / per-forward code (attention forward, cross-entropy, optimizer grad reduction, data-loader per-sample) where the redundant copy scales with a large dimension (vocab/tp, [seq,batch,vocab], attention [b,heads,sq,sk], GQA head expansion).
- GPU/CUDA (and ROCm/XPU) execution where CPU-allocated buffers incur host→device memcpy + potential host/device sync, especially under CUDA-graph capture sensitivity.
- Distributed / parallel setups: tensor-parallel (vocab-parallel), distributed optimizer (ZeRO grad/param buffer aliasing), pipeline-parallel comm buffers, MoE/expert-parallel, MLA attention.

**检测时该查什么（父快照核验）**：
- Is this on a hot path (per-forward/per-layer/per-step or per-sample in the data loader)? Does the copied tensor scale with a large dim (vocab, seq*batch, attention scores, expert params)?
- Is the op out-of-place (`a - b`, `a * b`, `x + bias`) where an in-place `_` variant is safe given the grad/eval mode? Check whether the caller is eval/no-grad or the tensor is a throwaway.
- Is a buffer allocated on CPU then `.cuda()`/`.type(...).cuda()`/moved to device? Replace with `device=current_device()` to avoid host alloc + H2D memcpy and sync.
- Is `.contiguous()`/`.clone()`/`.float()`/`.astype()` applied unconditionally? Verify the downstream consumer actually needs it (rotary applied? kernel requires contiguous? dtype must widen?).
- Does a kernel write to a temp then `out.copy_(tmp)`, or is an op stuck in an auto_functionalized wrapper preventing in-place mutation? Can the destination be passed directly?
- Do buffers that should alias the same storage (param_buffer vs grad bucket) actually share `data_ptr()`? Are collectives using flat-buffer variants instead of list-of-views? Does a `split`/`repeat`/`concat(list(...))` create avoidable copies?

**高发文件 top5**：`megatron/**/transformer/attention.py, core/**/attention (q/k/v contiguous, GQA repeat, value.contiguous, QKV split)`, `megatron/**/tensor_parallel/cross_entropy.py (vocab-parallel logits clone / max-subtract)`, `megatron/**/optimizer/distrib_optimizer.py & core/**/distributed (param/grad buffers, reduce-scatter/all-gather)`, `megatron/**/data/indexed_dataset.py (MMapIndexedDataset astype), forward_step / schedules (pipeline communicate, CPU→GPU tensor alloc)`, `vllm/model_executor/models/qwen*_vl.py & multimodal preprocessing, vllm/**/fusion passes (FixFunctionalizationPass), MLA attention backend`

### shared-workspace-reuse  （30 张 train 卡）

**典型反模式**：
- Allocating fresh scratch/intermediate tensors inside a per-call hot path (e.g. `torch.empty(...)` / `at::empty(...)` for all-gather buffers, MoE intermediate caches, pinned CPU input tensors, or GEMM workspaces) on every forward/backward step instead of reusing a persistent, resizable buffer.
- Constructing an expensive stateful object (CUDA Stream, RotaryEmbedding with cos/sin cache, VocabParallelEmbedding, static graph input buffers) once per layer/instance in a loop, instead of hoisting to a shared/module-level singleton or cache — duplicated N times where N = number of layers/microbatches/backend instances.
- Two independent allocations for tensors that are used disjointly (e.g. intermediate_cache1 and intermediate_cache3) when a single buffer sized `max(a,b)` plus slicing/aliasing would suffice; likewise separate CUDA-graph memory pools per graph/batch-size instead of one shared `graph_pool_handle`.
- Reallocating a persistent workspace on any size mismatch (`if _size != requested: free+malloc`) rather than treating it as a high-water-mark (`if _size < requested: grow`), causing churn when sizes vary across calls/layers.
- Materializing a transient full-model copy (e.g. `param.data.float()` for norm computation) instead of reusing an already-existing master/main copy.

**显现条件 top3**：
- Pipeline / virtual-pipeline / interleaved parallelism with CUDA graphs enabled — per-(layer, microbatch) static buffers and per-graph memory pools multiply; benefit scales with num_microbatches and VP degree.
- MoE / fused-experts paths and quantized (FP8/FP4/w8a8 Marlin) GEMMs where per-call intermediate caches and GEMM workspaces are freshly allocated each forward step.
- Per-layer construction of shared objects (rotary embeddings, CUDA streams, embeddings) so redundant caches/streams accumulate proportional to number of attention layers or backend instances.

**历史量级样本**：
- ~1GB freed (vllm EAGLE draft-model embedding skip for Llama 3 target when PP world_size==1)

**检测时该查什么（父快照核验）**：
- Is the allocation on a per-step / per-forward / per-backward hot path (e.g. inside `_prepare_inputs`, `forward`, attention/MoE kernel wrappers) rather than a one-time init?
- Is an object/buffer being created once per layer/microbatch/graph/instance where a single shared instance (module-level cache, GlobalMemoryBuffer, graph_pool_handle, main_param) would serve all callers with identical parameters?
- Does the workspace/buffer reallocate on exact size mismatch instead of growing to a high-water-mark? Check the guard (`!=` vs `<`).
- For paired disjoint scratch tensors, could they share one `max(...)`-sized buffer via slicing/aliasing?
- Does a new code path (e.g. an 'mrope' / custom-layer branch) bypass an existing cache lookup or reuse-eligibility check that the default path honors?
- Is the buffer keyed correctly on (shape, dtype) so reuse is safe, and does buffer-reuse activation depend on a flag/condition (e.g. reuse_input_output_buffer, PP world_size==1) that custom subclasses may not satisfy?

**高发文件 top5**：`Megatron-LM: transformer/cuda_graphs.py, model/*/transformer_layer static-input helpers, tensor_parallel/layers.py (GlobalMemoryBuffer / all_gather_buffer), optimizer/ (main_param, calc_params_l2_norm)`, `vllm: v1/worker/gpu_model_runner.py (_prepare_inputs, intermediate_tensors), model_executor/layers/fused_moe/* (fused_experts intermediate caches, marlin workspace)`, `vllm: model_executor/layers/rotary_embedding.py (get_rope / _ROPE_DICT), compilation/backends (VllmBackend graph_pool / global_graph_pool)`, `DeepSpeed: csrc transformer kernels (Context::GenWorkSpace, ds_linear_layer workspace allocation)`, `TransformerEngine: pytorch/graph.py (CUDA-graph static input/grad buffer reuse), module/grouped_linear.py (wgrad placeholder buffer)`

### transient-realloc-spike  （7 张 train 卡）

**典型反模式**：
- Merging/concatenating weight shards with `torch.cat([...])` (e.g. gated w+v gate merge) that materializes a second full-size buffer alongside the originals before the source tensors are freed.
- Growing an in-place buffer with `tensor.resize_(larger)` or appending KV state via `torch.cat((past, cur), dim=...)` per step/layer — the new (larger) allocation coexists with the old one, producing a transient 2x spike.
- Cross-device master->working copy via `dst.copy_(cpu_fp32_master.to(gpu).data)`: the intermediate `.to(device)` materializes a full-precision GPU tensor (often 2x element size) that briefly stacks on top of existing params.
- Materializing an oversized intermediate dtype before reduction, e.g. `(x > vals[:, None]).long().sum(1)` allocates a full (tokens, vocab) int64 tensor; and in-place random init `param.uniform_(...)` directly on the accelerator device instead of generating on CPU and copying over.
- Moving/casting a whole module to device (`module.to(cur_device)`) before sharding/injection, so the full unpartitioned model transiently occupies one GPU.

**显现条件 top3**：
- Near memory limit on the accelerator (CUDA GPU / TPU HBM); the extra transient buffer tips an otherwise-fitting workload into OOM.
- Large tensor dimensions amplify the spike — big vocab_size × tokens, full-size weight/KV shards, or full-precision (fp32) master copies of fp16/bf16 params.
- Model/tensor/pipeline parallel or offload paths (DCP checkpoint load, ZeRO stage 1/2 CPU offload, multi-GPU injection, DBO workspace) where a whole shard or full module is briefly held in addition to steady-state memory.

**历史量级样本**：
- ZeRO stage 1/2 CPU offload param update: ~3x params_FP16 transient before fix -> ~1x params_FP16 after (commit message).

**检测时该查什么（父快照核验）**：
- Does the operation allocate a new full-size buffer before the old/source one is freed? Look for `torch.cat`, `resize_` to larger, and `.to(device)`/`.copy_(x.to(device))` on large tensors.
- Is it on a hot / repeated path (per-token, per-layer, per-step, per-checkpoint-load)? Repetition and near-limit memory turn a transient spike into OOM.
- Is the intermediate dtype/precision larger than needed (int64 mask from `.long()`, fp32 master before working dtype)? Can the reduction/copy be done without materializing it?
- Is a whole module/tensor moved to a single device before sharding/injection/partitioning, so the unpartitioned form transiently sits on one GPU?
- Is there a memory-limit-adjacent config (offload, TPU, large vocab, big MLP shards) where the extra 1x–2x buffer matters? Is there a fallback/try-except for the OOM, or an in-place / CPU-staging alternative?

**高发文件 top5**：`megatron/**/transformer/mlp.py (gated/SwiGLU weight merge on checkpoint load)`, `megatron/**/attention or transformer/*.py (KV-cache / layer_past decode path)`, `vllm/**/workspace or attention backend (WorkspaceManager._ensure_workspace_size); vllm/**/sampler / model_executor (_get_ranks); vllm/**/model_loader (dummy weight init / TPU)`, `deepspeed/runtime/zero/**/stage_1_and_2.py (CPU-offload param update copy)`, `deepspeed/inference/**/engine.py (InferenceEngine.__init__ module.to/inject ordering)`

### uvm-managed-pagefault  （9 张 train 卡）

**典型反模式**：
- Enabling CUDA unified/managed memory (cudaMallocManaged) as the *default* allocation path for a large hot buffer (e.g. KV-cache memory_buffer) so it silently oversubscribes GPU memory; when the working set exceeds VRAM the buffer pages fault-migrate over PCIe on the critical path, trading a hard OOM for a throughput/latency cliff.
- Wrapping a performance-critical allocation in `torch.cuda.use_mem_pool` backed by a CUDAPluggableAllocator over managed memory without a runtime guard that the pages actually stay resident on-device — relying on cudaMemAdvise (SetPreferredLocation/SetAccessedBy) hints that are advisory and may be ignored by newer CUDA toolkits.
- Flipping a `unified_memory_level`-style default from 0 (on-GPU only) to 1 (UVM) in a shared config/example so every consumer inherits over-subscription semantics, even those whose buffer fits in VRAM and would run faster on plain GPU allocation.
- Restructuring an allocator from an explicit split/oversubscribe block model to a single UVM-backed buffer, coupling the paging policy to page-fault-driven migration rather than an explicit active/paused block accounting.

**显现条件 top3**：
- Dynamic-batching inference (DynamicInferenceContext) with the KV-cache buffer allocated in managed/unified memory and the working set exceeding on-device VRAM, so accesses trigger fault-driven host<->device migration.
- UVM/managed path enabled by default (unified_memory_level > 0) on GPUs where managed-memory behavior is PCIe/hardware dependent — same code shows opposite sign depending on whether the buffer fits in VRAM.
- Build/runtime against a newer CUDA toolkit (>= 13) where cudaMemAdvise preferred-location/accessed-by hints intended to pin pages on-device behave differently, allowing pages to migrate away and page-fault.

**历史量级样本**：
- steady-state throughput ~82->88 tok/s (fp8, cuda_graphs golden values) when flipping UVM level 1->0
- ~67->70 tok/s (decode_graphs_only) on the same flip
- per-token latency 0.3596->0.3365 (golden_values_dev_dgx_h100.json)

**检测时该查什么（父快照核验）**：
- Is this buffer on the inference hot path (per-step KV-cache read/write)? UVM paging there directly costs steady-state tok/s and per-token latency.
- Check the default of the unified_memory_level knob (kwarg default AND CLI arg default AND example/config .sh files) — a 0->1 or 1->0 flip is the common trigger; confirm which direction and whether the buffer typically fits in VRAM.
- Confirm the allocator path: cudaMallocManaged + CUDAPluggableAllocator + torch.cuda.use_mem_pool — and whether there is a fallback to level 0 on UnifiedMemoryUnsupportedError.
- Verify cudaMemAdvise hints (SetPreferredLocation/SetAccessedBy) actually prevent migration on the target CUDA toolkit version; treat them as advisory, not guarantees.
- Compare golden throughput/latency numbers (fp8/bf16, cuda_graphs vs decode_graphs_only) across the config flip to catch a ~several-percent regression rather than a hard OOM.

**高发文件 top5**：`megatron/**/inference/**/dynamic_context.py`, `megatron/**/inference/**/unified_memory.py`, `examples/rl/model_configs/*.sh`, `megatron/**/arguments.py (inference-dynamic-batching-unified-memory-level default)`, `megatron/**/data/*gpt* (mmap sequence_lengths / build_sample_idx page-fault path)`


## 大类：parallelism-scheduling

### cpu-affinity-numa  （15 张 train 卡）

**典型反模式**：
- Assuming a rank's pinned CPUs all live on a single NUMA node: parsing one node string and calling numa_set_membind(single_node) (or numactl -m <node> only when cores exactly match one node), so on multi-NUMA / HBM-flat / SNC / fakenuma topologies the match fails and no memory binding is applied — allocations land on remote nodes.
- Leaving OMP thread affinity defaulted to 'all'/no-binding, letting a rank's OpenMP threads spread across all logical CPUs including remote NUMA nodes, incurring cross-NUMA memory traffic.
- Spawning one worker process per shard (TP world_size>1) without setting OMP_NUM_THREADS, so each process's OpenMP/torch intra-op pool defaults to the full host core count → N processes × N threads oversubscription and thread contention (amplified under container CPU quotas).
- Computing core binding independently of the process's actual affinity set (e.g. from os.cpu_count() split evenly) instead of honoring the affinity set by the launcher/numactl, so worker threads bind to cores the process cannot actually run on.
- Pinning threads via a thread-bind path without ensuring a compatible OpenMP/libgomp runtime is loaded (ACL vs PyTorch libgomp mismatch, ppc64le libgomp misbehavior), collapsing utilization to a single core.

**显现条件 top3**：
- Multi-NUMA-node host (multi-socket x86_64, AMD EPFF with >1 node/socket, ppc64le, ARM) where a rank's bound CPUs span more than one node or where topology is non-normal (HBM-flat, SNC, fakenuma) so single-node membind assumptions break.
- CPU-backend tensor/data parallelism with one process per rank/shard and no explicit OMP_NUM_THREADS, causing OpenMP oversubscription across ranks (worse under containerized CPU limits/throttling).
- Thread-bind enabled (VLLM_CPU_OMP_THREADS_BIND / --bind_cores_to_rank) on platforms with an incompatible OpenMP runtime (ppc64le libgomp, AArch64 ACL), collapsing to single-core utilization.

**历史量级样本**：
- ZenFlow overlap: master 1381.41ms -> zenflow_affinity 1216.65ms (Qwen2.5-3B, 50-step avg)
- CPUAdam roofline ~0.6s vs zero-offload 0.805s (issue-measured)

**检测时该查什么（父快照核验）**：
- At NUMA-setup code: does it assume all pinned CPUs map to exactly ONE NUMA node? Check whether membind/numactl -m is skipped or wrong when cores span >1 node or on HBM-flat/SNC/fakenuma topologies.
- Is memory binding chosen consistently with thread binding (interleave vs membind) when CPUs span multiple nodes? Verify which mask/policy is actually applied.
- For multiprocess/TP launch: is OMP_NUM_THREADS set per rank, or does each process inherit the full host core count? Check for oversubscription = num_procs × host_cores.
- Does the worker derive its core set from the process's real affinity mask, or independently from os.cpu_count()/even-split? Look for divergence from launcher/numactl-assigned cores.
- Is thread-binding default-enabled (auto)? If so, confirm the correct/compatible OpenMP (libgomp/ACL) runtime is loaded so binding doesn't collapse to one core.
- Check env-var/flag defaults (VLLM_CPU_OMP_THREADS_BIND=auto, VLLM_CPU_NUM_OF_RESERVED_CPU=0, bind_cores_to_rank) — do defaults break for world_size>1 or reserve no cores?
- Is this on a hot path (per-worker init: init_cpu_threads_env, launcher core-binding, optimizer offload worker binding)? Regression is silent (throughput/latency), detectability typically low.

**高发文件 top5**：`csrc/cpu/utils.cpp (init_cpu_threads_env, NUMA membind/interleave, sched_setaffinity)`, `vllm/**/cpu*/ worker & platform files (VLLM_CPU_OMP_THREADS_BIND handling, get_cpus_id_binding_based_on_numa_nodes*, ppc64le/ARM binding routines)`, `vllm/executor/multiproc*_executor.py (MultiprocessingGPUExecutor, OMP_NUM_THREADS setup per shard)`, `DeepSpeed launcher/runner (get_numactl_cmd, IMPIRunner, --bind_cores_to_rank numactl -C/-m)`, `DeepSpeed ZenFlow CPU-offload optimizer worker (core-binding vs process affinity, CPUADAM_CORE_START/END)`

### eplb-issues  （11 张 train 卡）

**典型反模式**：
- MoE method/model class hard-blocks EPLB with `raise NotImplementedError("EPLB not supported for ... yet")` (or an early-return guard) instead of wiring `enable_eplb`/`num_redundant_experts` through to FusedMoE — leaving quantized/backend-specific and per-model paths unable to load-balance at all.
- Default "linear"/contiguous expert placement in the expert-map computation (`determine_expert_map`): contiguous logical experts land on the same EP rank, so for grouped-expert models a hot expert group overloads one rank while others idle; no `round_robin` (global_idx % ep_size) option available.
- Uneven remainder distribution in expert-map: giving each non-last rank `floor(global_num_experts/ep_size)` experts and dumping ALL remainder experts on the last rank when `global_num_experts % ep_size != 0`, creating a permanent straggler rank.
- Running the EPLB rearrange (weight redistribution) synchronously on the inference critical path — blocking `torch.cuda.synchronize()` followed by a serial per-layer loop of `batch_isend_irecv` + `req.wait()` and weight copies — stalling forward progress.
- Executing the per-forward logical→physical expert mapping and load-metric recording inline in eager mode inside `select_experts`, adding per-step Python/host overhead on the hot path instead of extracting it into a compiled helper.
- New expert placement strategy (e.g. round_robin) silently falling back to "linear" for a backend (e.g. DeepEP low-latency all2all) because the backend never wired it up.

**显现条件 top3**：
- Expert parallelism (ep_size>1) on multi-GPU/multi-node MoE deployments — imbalance/bubbles appear when token routing is skewed toward hot logical experts and every all-to-all/MoE step waits on the straggler rank.
- EPLB enabled (`enable_eplb=True`, often with `num_redundant_experts>0`) — either fully blocked by a NotImplementedError guard on a given quant method/model/backend, or paying rearrange/mapping overhead on the critical path.
- `global_num_experts % ep_size != 0`, or grouped-expert models (`num_expert_group>1`) under default linear placement — structural load imbalance across EP ranks.

**检测时该查什么（父快照核验）**：
- Is EPLB wiring reachable for this MoE method/model/backend, or is it gated by a `raise NotImplementedError(... EPLB not supported ...)` guard? (Check quantized MoE apply paths and per-model MixtureOfExperts implementations.)
- Is `enable_eplb`/`num_redundant_experts` actually plumbed down to FusedMoE (set_eplb_state / update_physical_experts_metadata / expert_weights registration), or dropped mid-plumbing?
- Which expert placement strategy is active in `determine_expert_map` — linear (contiguous) vs round_robin? Does the chosen backend actually honor it or silently fall back?
- Does `global_num_experts` divide evenly by `ep_size`? If not, how is the remainder distributed — evenly or all onto the last rank?
- Does the EPLB rearrange run on the inference critical path (blocking synchronize + serial P2P per layer), or is it overlapped/async?
- Is the per-forward logical→physical mapping + load-metric recording done inline in eager mode inside the hot select_experts path, or extracted/compiled?

**高发文件 top5**：`vllm/model_executor/layers/fused_moe/ (FusedMoE, select_experts, EPLB map/record, expert_map)`, `vllm/model_executor/layers/fused_moe/ (determine_expert_map / expert placement: linear vs round_robin)`, `vllm/model_executor/layers/quantization/ (quantized MoE methods: NVFP4/FP8, CompressedTensors WNA16 apply/EPLB guards)`, `vllm/model_executor/models/ (per-model MoE: Mixtral, Qwen3 MoE, Qwen3VLMoe, Transformers backend; MixtureOfExperts interface wiring)`, `vllm/distributed/ (EPLB rearrange_expert_weights_inplace, P2P batch_isend_irecv, DeepEP low-latency all2all backend)`

### fsdp-zero-pipeline-break  （9 张 train 卡）

**典型反模式**：
- Constructing full-size, PP-stage-specific modules (e.g. lm_head / vocab projection) unconditionally on every pipeline rank instead of guarding with `if is_last_rank`/`if is_first_rank`, wasting replicated weight memory on stages that never use them.
- Hardcoding pipeline concurrency knobs (e.g. `max_concurrent_batches` returning a constant `1`, or `num_warmup_microbatches == num_microbatches`) that force a synchronous GPipe-style all-forward-then-all-backward schedule and serialize execution, leaving pipeline bubbles and idle GPUs.
- Forcing a parameter class onto a replicated/non-distributed optimizer path (e.g. MoE expert params) with manual grad all-reduce, instead of letting the distributed/ZeRO optimizer shard their optimizer state and gradients across the appropriate parallel group.
- Computing pipeline tensor shapes globally/uniformly rather than per-stage, so sequence-parallel or encoder-decoder shape divisions (seq_length / TP, per-rank shapes) are wrong or hard-guarded out with a RuntimeError that blocks composition of parallelism modes.
- Resetting/advancing prefetch or trace step counters only at end-of-forward-pass rather than per-step, so sub-modules without their own parameters desynchronize the step_id and silently disable prefetching.
- Relying on activation/intermediate tensors saved via save_for_backward inside autograd Functions while under FSDP, where these module-internal saved activations are not sharded and occupy full unsharded memory.

**显现条件 top3**：
- Pipeline parallelism enabled (pp_size/pp_world_size > 1), especially with interleaved/virtual pipeline (VPP) or encoder-decoder model types; benefit/regression grows with number of microbatches.
- ZeRO / distributed optimizer or FSDP sharding active (ZeRO-1 for MoE expert params, ZeRO-3 prefetching, PyTorch FSDP with use_orig_params); sharding fails to cover certain params or saved activations.
- Composition of multiple parallelism modes (tensor + pipeline + sequence parallelism, or MoE expert-model-parallel + data parallel) where per-stage shapes/groups must be computed correctly.

**历史量级样本**：
- DeepSeek V3 lm_head ~129k vocab x hidden_size replicated weight on every non-last PP rank (order given, exact bytes n/a)

**检测时该查什么（父快照核验）**：
- At the parent snapshot, check whether stage-specific modules (lm_head, embeddings, vocab projection) are built on every PP rank vs. guarded by first/last-rank checks.
- Inspect the pipeline schedule: is it GPipe-style (all-forward-then-all-backward, num_warmup == num_microbatches) or 1F1B? Are concurrency knobs (max_concurrent_batches) hardcoded to 1?
- Verify tensor-shape computation is per-stage/per-rank (accounts for seq_length/TP division, encoder-decoder splits) rather than a single global shape or a hard RuntimeError guard.
- Check whether MoE/expert or other special params are routed to a replicated optimizer with manual all-reduce instead of the distributed/ZeRO optimizer's proper parallel group.
- For ZeRO-3 prefetch/trace: confirm step counters reset per-step and handle sub-modules without parameters; is prefetching silently disabled ('Tracing failed')?
- For FSDP: check whether autograd Functions save large intermediate/activation tensors that FSDP cannot shard, and whether activation recompute is disabled in that path.

**高发文件 top5**：`Megatron-LM: megatron/core/pipeline_parallel/schedules.py (get_tensor_shapes, forward_backward_pipelining_with_interleaving)`, `Megatron-LM: megatron/core/optimizer/ (distributed optimizer, MoE expert-param sharding)`, `vllm: model_defs for DeepSeek V2/V3/R1 (ParallelLMHead construction under PP)`, `vllm: v1 executor / engine core (RayDistributedExecutor max_concurrent_batches, step_with_batch_queue)`, `DeepSpeed: runtime/zero/ (PartitionedParameterCoordinator, PrefetchCoordinator) ; TransformerEngine: pytorch autograd Function modules (save_for_backward paths)`

### pipeline-stage-imbalance  （12 张 train 卡）

**典型反模式**：
- Even-split layer partitioning that dumps the remainder (`num_layers % pp_size`) onto the last (or first) pipeline stage, so `end_layer = num_layers` for the final rank — the overloaded stage becomes the bubble/throughput bottleneck.
- Layer-count math that ignores non-transformer layers (embedding, loss/lm_head, MTP): the stage owning these gets systematically more work than a pure decoder-layer count implies, because the split does not `account_for_embedding/loss_in_pipeline_split`.
- Balanced-partition routines (binary-search `_lprobe`/`_rb_partition_balanced`, or `chunksize = floor(num_items/num_parts)` with residual on the last part) that return badly skewed per-stage assignments instead of near-even ones.
- Non-1F1B / naive pipeline schedules that run ALL microbatch forwards then ALL backwards, or that omit interleaving (virtual pipeline chunks) so idle stages sit in fill/drain bubbles.
- Sharding all modules by a factor that does not divide evenly (e.g. `num_kv_heads % mp_size != 0`) so per-rank shard sizes are uneven, imbalancing compute across parallel ranks.

**显现条件 top3**：
- Pipeline parallel with PP>1 and num_layers not divisible by pp_size — remainder concentrated on one stage.
- Imbalanced models (embedding/loss/MTP layers) with VPP enabled but layout not accounting for those layers.
- num_microbatches ≤ pipeline_parallel_size (or micro_batch_size ≥ batch_size in inference), so fill/drain bubbles dominate.

**检测时该查什么（父快照核验）**：
- At the layer-partition function (get_pp_indices / get_num_layers_to_build / partition_uniform / partition_balanced): does the last or first rank absorb the entire remainder? Prefer near-even distribution.
- Are non-decoder layers (embedding, loss, lm_head, MTP) counted toward per-stage budgets, or silently piled onto the boundary stages?
- Is the schedule 1F1B / interleaved (VPP), or naive all-forward-then-all-backward? Check num_microbatches vs pipeline_parallel_size for bubble ratio.
- Is there a manual override (env var / layout string) available, and does the default path stay balanced when the override is unset?
- For sharding-based imbalance: does the shard-size divisor evenly divide the dimension for every module (kv_heads vs mp_size), or do some modules need near-even division?
- With CUDA graphs + fine-grained/interleaved scheduling: are wgrad/dgrad graph captures ordered so no stage stalls (delay_wgrad_compute coupling)?

**高发文件 top5**：`Megatron-LM: megatron/core/transformer/ (get_num_layers_to_build, TransformerLayer._get_layer_offset), pipeline layout config`, `Megatron-LM: megatron/core/pipeline_parallel/schedules.py (1F1B / interleaved forward_backward_pipelining)`, `vllm: get_pp_indices layer-partition utility (distributed/pp utils)`, `DeepSpeed: runtime/pipe/module.py & topology (PipelineModule._partition_layers, partition_uniform/partition_balanced)`, `DeepSpeed: AutoTP get_shard_size (module sharding); TransformerEngine: CUDA-graph capture path for delay_wgrad_compute`

### reshard-tradeoff  （27 张 train 卡）

**典型反模式**：
- Choosing full-domain sharding (ZeRO-3 / reshard_after_forward=True) as the unconditional default, forcing a re-all-gather of parameters before every backward when the memory savings aren't needed — the recomputed all-gather is pure comm overhead on the hot path.
- Sharding optimizer/param/grad state across the *full* DP(+CP) domain when a partial/hybrid sub-group (intra-instance shard + inter-instance all-reduce) would trade a small extra all-reduce for far larger reduce-scatter/all-gather collective sizes.
- Recursively installing per-submodule gather/re-partition hooks on fine-grained or MoE modules, so the parameter all-gather+partition fires per-expert/per-token with data-dependent execution order (disrupts prefetch, adds per-hook overhead, can deadlock the collective).
- Re-gathering/re-partitioning weights on every forward during autoregressive generation or repeated inference passes instead of gathering once and holding them for the loop.
- Duplicating/replicating a small (low-rank) projection across TP ranks under a 'duplicated' parallel mode where a column/row-parallel sharded layout would avoid redundant compute and the associated gather path.
- Unconditionally partitioning pipeline activation/grad buffers across the model-parallel group, saving memory but forcing extra all-gather collectives that hurt throughput when memory is not the bottleneck.

**显现条件 top3**：
- ZeRO-3 / full-domain FSDP sharding vs. a partial/hybrid/ZeRO-2 alternative: default reshard-after-forward or full-DP-domain sharding pays comm cost (re-all-gather, larger reduce-scatter) that a partial-instance or reshard=False mode would avoid.
- MoE / fine-grained nested modules under param-partitioned sharding (ZeRO-3, expert/context parallel): per-submodule hook gather+partition with data-dependent expert order causes bubbles, hook overhead, or all-gather deadlock/hang.
- Repeated forward passes (autoregressive generation, per-token inference) under partitioned params: gather/partition cost is paid every pass instead of amortized once.

**检测时该查什么（父快照核验）**：
- At the parent snapshot: is the sharding domain the FULL DP(+CP) group, or a partial/hybrid sub-group? A whole-domain reduce-scatter/all-gather trades throughput for memory — check whether a partial-instance (intra shard + inter all-reduce) path is available.
- Is reshard-after-forward / ZeRO-3 the DEFAULT? If so, parameters are re-all-gathered before backward — confirm whether ZeRO-2 (reshard=False) or param-persistence for small params is the better tradeoff for the workload.
- Is this a hot path — per-layer forward/backward hooks, per-expert MoE dispatch, or per-token generation loop? Recursive hook registration and per-pass gather multiply the collective count.
- For MoE / fine-grained modules: are gather hooks registered recursively on children, or is there a z3-leaf / granularity-threshold coalescing that stops recursion? Absence implies per-submodule all-gather overhead and possible deadlock.
- For small/low-rank projections: is the layer sharded (column/row-parallel) or replicated ('duplicated' mode)? Replication trades comm for redundant compute across TP ranks.
- Is the memory-saving buffer partitioning (checkpointed activations, pipeline act/grad buffers) unconditional, and does it add all-gather collectives on the critical path even when memory headroom exists?

**高发文件 top5**：`Megatron-LM: megatron/core/distributed/ (custom_fsdp / param_and_grad_buffer, distributed_optimizer, torch_fsdp2 config & fully_shard plumbing)`, `Megatron-LM: megatron/core/optimizer/ & optimizer config (num_distributed_optimizer_instances, partial-distopt group setup)`, `DeepSpeed: deepspeed/runtime/zero/ (parameter_offload.py, partition_parameters.py — z3 leaf modules, ds_persist, hook registration)`, `DeepSpeed: deepspeed/runtime/pipe/ (pipeline activation/grad buffer partitioning under model-parallel)`, `vllm: vllm/model_executor/layers/ (fused_moe / expert-parallel, *ParallelLinear for TP; vllm/lora/fully_sharded_layers.py)`

### scheduler-work-distribution  （19 张 train 卡）

**典型反模式**：
- Head-of-line blocking in the scheduler's waiting/queue loop: on encountering a request that can't be scheduled (LoRA slot full, partial-state present, over budget), doing `break` for the entire loop instead of `continue`, so unrelated schedulable requests are starved for the step.
- Greedy load-balancer / bin-packing that routes to the least-occupied worker but seeds per-worker counters wrong (e.g. initializing load array to empty `[]` with an `if not counts` early-return, or sorting shards ascending instead of descending for LPT), collapsing all work onto one rank.
- Passing a scalar sized for the common case into a work-splitting / tile-scheduler metadata function (e.g. num_q_heads assumed q_len=1), so SM/thread work distribution is miscomputed once query length > 1 (spec-decode/MTP).
- Parallelizing over the coarse dimension (one task per expert / per param) rather than the flattened fine-grained assignment (token-expert pairs / round-robin params), leaving thread pool or DP ranks idle and unbalanced.
- Hardcoding thread-pool / worker parallelism (max_workers=2, chunked Pool.map that barriers each batch) instead of sizing to available cores and streaming submissions, serializing independent work.

**显现条件 top3**：
- Data-parallel / expert-parallel deployments (DP>1, internal LB, distributed ckpt save).
- Heterogeneous concurrent request mix causing head-of-line blocking.
- Query length > 1 (spec-decode/MTP) breaking q_len=1 work-split assumptions.

**检测时该查什么（父快照核验）**：
- Is this on the per-step scheduler hot path? Check whether an unschedulable request triggers `break` (halts all remaining) vs `continue` (skips only that one).
- Are load-balancer/bin-packing counters and their initial state correct? Check empty-init guards (`if not counts`) and sort direction (ascending vs descending for LPT) that could funnel work to one rank.
- Do work-split / tile-scheduler args assume the common case? Verify the value passed for work distribution (e.g. tokens-per-head, num tasks) holds when q_len>1 or workload shape changes.
- Is the parallelization dimension coarse (per-expert/per-param) leaving threads/ranks idle? Check for redundant idle ranks during checkpoint save or unbalanced gradient/copy ownership.
- Is worker/thread-pool concurrency hardcoded or barriered per-batch instead of sized to cores and streamed?
- Is the feature default-enabled? Note opt-in flags (fully_parallel_save, enable_chunked_prefill + max_num_partial_prefills>1, num_scheduler_steps>1, enable-lora) — regression only manifests when enabled.

**高发文件 top5**：`vllm/v1/core/**/scheduler*.py (WAITING/RUNNING loop, partial-prefill, LoRA slotting)`, `vllm/v1/engine/**/*dp*client*.py (DPLBAsyncMPClient / DPAsyncMPClient load balancing & routing)`, `megatron/**/dist_checkpointing/**/fully_parallel*.py (FullyParallelSaveStrategyWrapper, distribute_chunks_to_ranks)`, `**/attention/**/*mla*.py (MLA decode metadata builder, get_mla_metadata tile-scheduler args)`, `deepspeed/**/checkpoint/**convert*.py & zero optimizer (round-robin gradient reorder, Pool.map chunking)`

