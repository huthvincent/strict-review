# Stage D2 — 全量打标报告

- generated: 2026-07-18T06:05:01.723217+00:00 · model us.anthropic.claude-opus-4-8 · 累计 ~$84.60
- 打标 perf 卡: 5016 / 5016
- other-unclear: 1 (0.0%) ✓ <5%
- 10% 双标: 500 卡 · 一致率 97.2% · **Cohen's κ (leaf) = 0.971** ✓ ≥0.6

## 叶子分布(top 25)

| leaf | n |
|---|--:|
| suboptimal-kernel-gate | 398 |
| kernel-fusion | 323 |
| fused-backend-swap | 307 |
| redundant-hot-path-work | 305 |
| kernel-config-tuning | 209 |
| config-toggle-perf-feature | 200 |
| cudagraph-enablement | 173 |
| boolean-guard-misfire | 136 |
| blocking-d2h-sync | 136 |
| redundant-buffer-copy | 129 |
| missing-comm-overlap | 116 |
| buffer-sizing-layout | 114 |
| kernel-occupancy-redesign | 110 |
| memory-leak-unpruned-state | 109 |
| collective-payload-reduction | 101 |
| wasted-compilation-work | 92 |
| low-precision-state | 89 |
| redundant-collective | 84 |
| host-scalar-loop-work | 80 |
| cudagraph-incompatibility | 79 |
| mismatched-collective-config | 74 |
| offload-management | 74 |
| redundant-load-startup-work | 73 |
| warmup-shape-gap | 71 |
| prefix-state-caching | 69 |
