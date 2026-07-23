# Stage D1 — 分类学(consolidated,待 Rui 过目)

- generated: 2026-07-18T05:43:10.753472+00:00 · model us.anthropic.claude-opus-4-8 · consolidation cost ~$1.33
- raw leaves 256 → **canonical 74 叶 / 11 大类** (目标 25–60)
- raw→canonical 覆盖: 256/256 ✓ ⚠️ 无效目标 6

## 分类学(大类 → 叶子)

### Collective & Communication (`collective-comm`, 6 叶)
- **redundant-collective** — Redundant/Duplicate Collective: 
- **mismatched-collective-config** — Mismatched Collective Shape/Group/Config: 
- **collective-payload-reduction** — Collective Payload Reduction: 
- **missing-comm-overlap** — Missing Compute-Comm Overlap: 
- **ipc-transport-overhead** — IPC/Transport Path Overhead: 
- **rank-placement-locality** — Rank Placement / Comm Locality: 

### Concurrency & Synchronization (`concurrency-sync`, 8 叶)
- **stream-serialization** — Single-Stream/Over-Broad Serialization: 
- **defensive-oversync** — Defensive Over-Synchronization: 
- **blocking-d2h-sync** — Blocking Device-to-Host Sync: 
- **host-scalar-loop-work** — Host-Side Scalar/Loop Work On GPU Tensors: 
- **gil-held-blocking-call** — Blocking Call Holding GIL: 
- **spin-wait-hang** — Spin-Wait / Counter-Overflow Hang: 
- **shared-mutable-state-contention** — Shared Mutable State Contention: 
- **sm-margin-reservation** — SM-Margin Reservation For Overlap: 

### Host-Side & Dispatch Overhead (`host-overhead`, 5 叶)
- **redundant-hot-path-work** — Redundant Hot-Path Reshape/Alloc/Metadata: 
- **one-time-setup-on-hot-path** — One-Time Setup On Hot Path: 
- **quadratic-python-structure** — Quadratic/Churny Python Data Structures: 
- **eager-error-string-build** — Eager Error-Message/Payload Construction: 
- **custom-op-dispatch-overhead** — Op Wrapper/Dispatch Overhead: 

### Kernel Efficiency (`kernel-efficiency`, 10 叶)
- **kernel-fusion** — Kernel Fusion: 
- **fused-backend-swap** — Swap To Fused/Specialized Backend: 
- **unfused-transpose-copy** — Unfused Transpose/Contiguous Copies: 
- **kernel-occupancy-redesign** — Under-Parallelized Kernel Redesign: 
- **kernel-config-tuning** — Kernel Launch-Config/Tile Tuning: 
- **suboptimal-kernel-gate** — Suboptimal/Over-Restrictive Kernel Gate: 
- **kernel-noop-skip** — Skip Kernel Work For No-Op Config: 
- **algorithmic-compute-reduction** — Algorithmic Compute Reduction: 
- **correctness-forces-slow-path** — Correctness/Determinism Forces Slow Path: 
- **dma-access-order-relax** — Relax DMA Access-Order Constraint: 

### Compilation & Graph Capture (`compilation`, 6 叶)
- **dynamo-graph-break** — Graph Break / Compile-Blocking Construct: 
- **cudagraph-enablement** — CUDA-Graph Capture/Replay Enablement: 
- **cudagraph-incompatibility** — CUDA-Graph Capture Incompatibility: 
- **wasted-compilation-work** — Wasted/Over-Broad Compilation Work: 
- **compile-cache-misconfig** — Compile-Cache Key/Write Misconfiguration: 
- **warmup-shape-gap** — JIT Warmup Shape/Coverage Gap: 

### Memory Management (`memory-management`, 8 叶)
- **deferred-free-peak** — Deferred Free / Retained Buffer: 
- **memory-leak-unpruned-state** — Unpruned/Leaked Tracking State: 
- **redundant-buffer-copy** — Redundant Buffer Copy/Zeroing: 
- **buffer-sizing-layout** — Wrong-Scope/Oversized/Layout Buffer: 
- **transient-realloc-spike** — Transient Realloc/Growth Spike: 
- **shared-workspace-reuse** — Shared/Reused Workspace Buffer: 
- **allocator-config** — Allocator/Memory-Pool Configuration: 
- **uvm-managed-pagefault** — UVM/Managed-Memory Page-Fault Overhead: 

### Memory Footprint Reduction (`memory-footprint`, 5 叶)
- **activation-recompute** — Activation Recomputation/Checkpointing: 
- **offload-management** — Activation/Weight/KV Offload Management: 
- **low-precision-state** — Low-Precision State Storage: 
- **skip-saved-tensor-backward** — Skip Unneeded Saved-For-Backward Tensor: 
- **meta-device-init** — Meta-Device / Deferred Parameter Init: 

### Parallelism & Scheduling (`parallelism-scheduling`, 6 叶)
- **reshard-tradeoff** — Reshard/Sharding Memory-Comm Tradeoff: 
- **fsdp-zero-pipeline-break** — FSDP/ZeRO Prefetch/Release Path Break: 
- **pipeline-stage-imbalance** — Pipeline Stage Layer Imbalance: 
- **eplb-issues** — Expert-Parallel Load-Balancing Defects: 
- **scheduler-work-distribution** — Conservative Scheduler / Poor Work Binning: 
- **cpu-affinity-numa** — CPU Affinity / NUMA Binding: 

### I/O, Loading & Startup (`io-startup`, 7 叶)
- **serial-io** — Serialized/Single-Rank/GIL-Bound I/O: 
- **async-io-overlap** — Async I/O Overlap For Offload/Checkpoint: 
- **slow-fs-path** — Slow Filesystem / Bounce-Buffer Path: 
- **weight-load-strategy** — Weight-Load Strategy/Format: 
- **redundant-load-startup-work** — Redundant Load-Time/Startup Work: 
- **process-launch-overhead** — Process Launch/Bootstrap Overhead: 
- **weight-transfer-sync** — Pipelined/Chunked Weight Transfer & Sync: 

### Inference & Serving Strategy (`inference-serving`, 7 叶)
- **prefix-state-caching** — Prefix/State Caching Reuse: 
- **prefill-decode-kernel-split** — Prefill/Decode & Dense/Sparse Path Split: 
- **speculative-decoding** — Speculative Decoding Strategy: 
- **encoder-execution-mode** — Multimodal Encoder Execution Mode: 
- **async-output-overlap** — Async Output D2H / Delta Serialization: 
- **kv-cache-capacity** — KV/Cache Capacity & Hold Tuning: 
- **sparse-layer-update** — Sparse/Skipped Layer Update Technique: 

### Configuration & Observability (`config-observability`, 6 叶)
- **config-toggle-perf-feature** — Config Default Toggle Of Perf Feature: 
- **boolean-guard-misfire** — Boolean Guard Misfires Fast Path: 
- **batch-composition-mismatch** — Batch-Composition Assumption Mismatch: 
- **wrong-index-dtype-hang** — Wrong Index Dtype Hang: 
- **failure-detection-gap** — Subprocess/Startup Failure Detection Gap: 
- **profiling-accounting-fix** — Profiling/Memory Accounting Fix: 

