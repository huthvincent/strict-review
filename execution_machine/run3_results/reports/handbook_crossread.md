# 手册 10 叶对读（语义版，§2.1③修正）

- generated: 2026-07-23T00:02:53.854459+00:00 · seed 20260720 · **语义比较**（LLM judge），非词面 token 重叠
- 前次用词面 token 重叠 → 10/10 误报 REWRITE（LLM 自由文本复现概念而非 token）。此为修正版。
- 含『外源内容检查』栏（§2.1③要求）：两次蒸馏是否出现无法由通用机制推出的具体标识符。

| 叶 | 概念重叠 | 需重写(>50%无对应)? | 外源内容? | 备注 |
|---|--:|:--:|:--:|---|
| meta-device-init | 0.9 | ✓ | ✓ 无 |  |
| missing-comm-overlap | 0.9 | ✓ | ✓ 无 |  |
| compile-cache-misconfig | 0.92 | ✓ | ✓ 无 |  |
| prefix-state-caching | 0.95 | ✓ | ✓ 无 |  |
| deferred-free-peak | 0.95 | ✓ | ✓ 无 |  |
| fused-backend-swap | 0.95 | ✓ | ✓ 无 |  |
| process-launch-overhead | 1.0 | ✓ | ✓ 无 |  |
| redundant-collective | 1.0 | ✓ | ✓ 无 |  |
| sm-margin-reservation | 1.0 | ✓ | ⚠️ 有 |  |
| offload-management | 1.0 | ✓ | ✓ 无 |  |

- 需重写: **0/10** · 外源内容: **1/10**
- 对 0 个需重写叶重蒸取交集；1 个外源信号交泄漏审计处理。
