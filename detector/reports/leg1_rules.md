# Leg 1 — static rules (metrics.v1 §3.1)

- generated: 2026-07-19T07:05:27.540999+00:00
- mined 285 drafts → 246 executable → **31 kept**, 254 rejected
- keep rule: dev precision ≥0.5 AND ≥2 dev-positive hits
- **dev FPR of kept ruleset: 2.6%** (target ≤5%)
- success gate (≥15 kept AND FPR≤5%): ✓ MET

## kept rules

| rule_id | leaf | prec | pos | neg | severity |
|---|---|--:|--:|--:|---|
| softmax-full-then-index-subset | redundant-hot-path-work | 1.0 | 2 | 0 | important |
| gpu-metadata-build-max-reduce-item-sync | blocking-d2h-sync | 1.0 | 2 | 0 | important |
| numpy-index-gpu-tensor-d2h-sync | blocking-d2h-sync | 1.0 | 3 | 0 | important |
| device-cumsum-into-zeros-buffer | host-scalar-loop-work | 1.0 | 3 | 0 | important |
| pin-memory-ignores-config-flag | boolean-guard-misfire | 1.0 | 6 | 0 | important |
| cuda-stream-in-init | shared-workspace-reuse | 0.929 | 13 | 1 | important |
| early-exit-misses-group-none-guard | redundant-hot-path-work | 0.857 | 6 | 1 | important |
| unconditional-device-copy-may-alias-cpu | redundant-buffer-copy | 0.769 | 20 | 6 | suggestion |
| async-busy-poll-sleep-zero | spin-wait-hang | 0.75 | 3 | 1 | important |
| item-in-accumulation-loop | blocking-d2h-sync | 0.75 | 3 | 1 | important |
| tensor-on-gpu-before-cpu-sync | blocking-d2h-sync | 0.714 | 5 | 2 | important |
| all-gather-object-consumed-only-on-rank0 | collective-payload-reduction | 0.667 | 4 | 2 | important |
| redundant-copy-of-kernel-output-buffer | redundant-buffer-copy | 0.667 | 6 | 3 | important |
| split-squeeze-contiguous-chunk | redundant-buffer-copy | 0.667 | 2 | 1 | important |
| tensor-item-in-accumulation-loop | blocking-d2h-sync | 0.667 | 2 | 1 | important |
| tensor-item-in-norm-accumulation-loop | blocking-d2h-sync | 0.667 | 2 | 1 | important |
| param-flatten-in-forward-hot-path | one-time-setup-on-hot-path | 0.667 | 2 | 1 | important |
| device-query-in-loop-config | redundant-hot-path-work | 0.667 | 2 | 1 | important |
| full-cuda-sync-in-overlap-path | defensive-oversync | 0.632 | 12 | 7 | important |
| all-gather-then-local-shard-split | redundant-collective | 0.579 | 11 | 8 | important |
| tensor-item-in-forward-arg | blocking-d2h-sync | 0.512 | 21 | 20 | important |
| tensor-from-pylist-on-device | blocking-d2h-sync | 0.511 | 23 | 22 | important |
| cpu-alloc-then-cuda-copy | redundant-buffer-copy | 0.5 | 4 | 4 | important |
| inverted-cuda-gate-selects-native-kernel | suboptimal-kernel-gate | 0.5 | 3 | 3 | important |
| no-cpu-in-serialization-hotpath | blocking-d2h-sync | 0.5 | 5 | 5 | important |
| tensor-cpu-alloc-then-to-device | redundant-buffer-copy | 0.5 | 7 | 7 | important |
| zeros-for-overwritten-workspace | redundant-hot-path-work | 0.5 | 2 | 2 | important |
| reassign-sorted-instead-of-inplace-sort | redundant-hot-path-work | 0.5 | 7 | 7 | suggestion |
| len-of-list-building-getter | redundant-hot-path-work | 0.5 | 2 | 2 | important |
| tensor-to-cpu-forces-d2h-sync | blocking-d2h-sync | 0.5 | 23 | 23 | important |
| tensor-cpu-numpy-roundtrip | blocking-d2h-sync | 0.5 | 5 | 5 | important |

rejected rules + reasons → `rules/rejected.jsonl` (负结果也是论文材料)
