# Stage 5 — 跨仓库泛化 (repo hold-out, protocol.v1 §1)

- generated: 2026-07-19T18:29:37.750574+00:00 · judge us.anthropic.claude-opus-4-8 (κ=0.926) · **static ceiling 57.1%**
- 设置：知识库(leg2 KB)与规则(leg1)**排除全部 Megatron 来源**（detector_v1_noMega），
  在 test 子集的 **Megatron 部分**（94 pos + 11 pairs + 175 neg = 280）上与全量知识 detector_v1 对比。
- leg3 recipe-map 本就是 Megatron-CI 专用（两者相同），故本实验隔离的是 leg1+leg2 的跨仓迁移。

## 结果（Megatron test part, 同一子集）

| 变体 | 知识库 | overall recall@2 | regression-fix@2 | benign FPR | leak |
|---|---|---|---|---|--:|
| detector_v1 | 全量(含Megatron) | 13/105 (12.4%) | 2/25 (8.0%) | 27/175 (15.4%) | 0 |
| detector_v1_noMega | 排除Megatron | 8/105 (7.6%) | 2/25 (8.0%) | 23/175 (13.1%) | 0 |

**跨仓迁移损失（overall recall@2）= +4.8pp**（正=全量知识更强）。

## 解读
- 该差值回答这套方法是否只对见过的仓库有效。leg1/leg2 排除 Megatron 后，
  在 Megatron 上的召回变化量化了知识迁移的依赖度。
- 注意 S4 消融已显示 leg1/leg2 在 test 上净贡献偏弱、leg3 承载主要召回；
  而 leg3 两版相同，故此处差值主要反映 leg1/leg2 的跨仓成分，量级预期较小。
