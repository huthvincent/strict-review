# Leg 1 — static rules (metrics.v1 §3.1)

- generated: 2026-07-19T18:16:01.022688+00:00
- mined 227 drafts → 198 executable → **22 kept**, 205 rejected
- keep rule: dev precision ≥0.5 AND ≥2 dev-positive hits
- **dev FPR of kept ruleset: 2.1%** (target ≤5%)
- success gate (≥15 kept AND FPR≤5%): ✓ MET

## kept rules

| rule_id | leaf | prec | pos | neg | severity |
|---|---|--:|--:|--:|---|
| gpu-metadata-build-reads-cpu-max-item | blocking-d2h-sync | 1.0 | 3 | 0 | important |
| cumsum-out-into-preallocated-gpu-zeros | host-scalar-loop-work | 1.0 | 3 | 0 | important |
| pin-memory-ignores-config-flag | boolean-guard-misfire | 1.0 | 3 | 0 | important |
| no-cpu-in-get-extra-state | blocking-d2h-sync | 1.0 | 6 | 0 | important |
| getenv-not-cached-in-function | one-time-setup-on-hot-path | 0.857 | 6 | 1 | important |
| redundant-copy-after-op-into-fresh-buffer | redundant-buffer-copy | 0.8 | 8 | 2 | important |
| unconditional-copy-to-device-tensor | redundant-buffer-copy | 0.769 | 20 | 6 | suggestion |
| busy-poll-loop-sleep-zero | spin-wait-hang | 0.75 | 3 | 1 | important |
| pybind-def-missing-gil-release | gil-held-blocking-call | 0.75 | 3 | 1 | important |
| accumulate-with-item-sync | blocking-d2h-sync | 0.667 | 2 | 1 | important |
| tensor-item-in-loop-accumulate | blocking-d2h-sync | 0.667 | 2 | 1 | important |
| cuda-sync-around-walltime-timer | blocking-d2h-sync | 0.632 | 12 | 7 | important |
| full-cuda-sync-in-overlap-path | defensive-oversync | 0.607 | 17 | 11 | important |
| unguarded-debug-fstring-in-hotpath | redundant-collective | 0.588 | 10 | 7 | suggestion |
| redundant-copy-of-kernel-return | redundant-buffer-copy | 0.524 | 22 | 20 | important |
| tensor-from-pylist-on-device | blocking-d2h-sync | 0.511 | 23 | 22 | important |
| tolist-in-comprehension-hot-path | blocking-d2h-sync | 0.5 | 2 | 2 | important |
| inverted-cuda-gate-on-fused-kernel | suboptimal-kernel-gate | 0.5 | 3 | 3 | important |
| reassign-sorted-of-self | redundant-hot-path-work | 0.5 | 7 | 7 | suggestion |
| eager-softmax-before-slicing | redundant-hot-path-work | 0.5 | 2 | 2 | suggestion |
| has-x-var-assigned-is-none | boolean-guard-misfire | 0.5 | 2 | 2 | important |
| tensor-cpu-numpy-roundtrip | blocking-d2h-sync | 0.5 | 5 | 5 | important |

rejected rules + reasons → `rules/rejected.jsonl` (负结果也是论文材料)
