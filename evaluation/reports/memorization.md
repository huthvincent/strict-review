# Stage 1.4 — Memorization 探测报告（全量）

- generated: 2026-07-19 · model: **裸 Opus 4.8**（无工具/无 diff/无上下文，仅 repo@sha + subject）· prompt memorization_probe.v1
- 探测目标: **934** = test perf-cards **818（全量，spec §1.4）** + test pairs **116**
- 成本 ~$6.4（初探 196 项 $1.35 + 全量补探 738 项 $5.06）

## 结果

- **memorized = true: 0 / 934**（0%）
- 模型声称"认得该 commit"的: **0**
- 声称 recall_confidence ≥ 0.5 的: **0**
- 其中给出可对 record 验证的具体事实（fix PR / 症状）的: **0**

## 判定口径（protocol §3）

memorized=true 需**同时**满足: `recall_confidence ≥ 0.5` **且**答案指认了真实可核对的事实
（真实 fix / culprit PR 号 或 正确症状）; 仅凭 subject 复述不算。

## 结论

裸模型无法从记忆复现这些 commit 的性能后果 → **test 集无记忆污染（全量确认）**。
主指标可在**全 test 集（818 正样本）**上报，无需退到 novel 子集；
即 spec §1.4 的 "novel 子集 = 全集" 成立。flags 为旁路标注，未改动 splits。

## 按 repo 分布（全量 934）

| repo | probed |
|---|--:|
| vllm | 740 |
| Megatron-LM | 105 |
| TransformerEngine | 68 |
| DeepSpeed | 21 |

旁路文件: `splits/memorization_flags.jsonl`（934 行，每行含 model_recall 原始输出 + grade_reason）。
初探快照留存于 `splits/memorization_flags.pre_full.jsonl`（196 行）。
