# Human Audit Packet — 抽样方法与计算口径 (Stage 7)

- 生成 2026-07-19T06:23:44.750803+00:00 · seed 20260719

## 抽样
- **A 组 360 卡**：从 5,016 张 perf 卡按 repo × kind × static_detectability 比例分层（最大余数法），再对不足 5 项的大类补足至 ≥5，覆盖全部 11 类。
- **B 组 150 对**：A 级 100 / B 级 50，按 repo 分层。

## 类别覆盖
| category | A组抽样数 |
|---|--:|
| collective-comm | 35 |
| compilation | 34 |
| concurrency-sync | 44 |
| config-observability | 21 |
| host-overhead | 26 |
| inference-serving | 20 |
| io-startup | 5 |
| kernel-efficiency | 111 |
| memory-footprint | 13 |
| memory-management | 39 |
| parallelism-scheduling | 12 |

## agreement 与 95% CI 计算口径
- 回填 `human_audit_template.jsonl` 后：
  - **per-field agreement** = 同意数 / (同意 + 不同意)，'无法判断' 从分母剔除并单列。
  - **Cohen's κ**（机器 vs 人）：对 kind/taxonomy 这类分类字段，按混淆矩阵计算。
  - **95% CI**：Wilson 区间（与主评测 FPR 同口径）。
- `apply_audits.py`（回填后运行）可消费该模板，产出 agreement / κ / CI。

> **本阶段只准备材料,不产生标签。** 与数据集'无人工真值'的诚实声明配套使用。
