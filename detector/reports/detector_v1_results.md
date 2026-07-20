# detector_v1 — test showdown + ablations (Stage 4, metrics.v1)

- generated: 2026-07-19T18:02:08.834612+00:00 · **static ceiling 57.1%** · **memorization 0/934 (novel = full test)**
- judge us.anthropic.claude-opus-4-8 (κ=0.926, cross-family Nova 89%) · test touched ONCE · leak_attempts 0
- full test subset = 1888 (818 perf-pos + 116 pairs + 954 neg); ablation subset = 1334 (§4.2: pos全量 + 116 pairs + 400 neg subsample)

## 主表 · recall@budget（full test subset）
> baseline (a)-(d) 的完整主表见 `reports/baseline_results.md`（同一冻结子集，同一 judge）。
> 下表列 detector_v1；关键对照数字在诚实小节汇总。

| detector | recall@1 | recall@2 | recall@5 | FPR(weighted, 95%CI) | leak |
|---|---|---|---|---|--:|
| **detector_v1** | 168/934 (18.0%) | 174/934 (18.6%) | 174/934 (18.6%) | 142/954 (14.9%, [12.8,17.3]) | 0 |

## 主表 · per-kind recall@2（**regression-fix = 北极星**）

| detector | **regression-fix** | optimization | config-default-change | perf-infra-or-test | not-perf | unclear |
|---|---|---|---|---|---|---|
| **detector_v1** | **32/261 (12.3%)** | 122/615 (19.8%) | 13/38 (34.2%) | 3/9 (33.3%) | 4/9 (44.4%) | 0/2 (0.0%) |

## 主表 · per-taxonomy recall@2（detector_v1 vs Opus baselines）

| category | detector_v1 | (a) Megatron | (b) generic |
|---|---|---|---|
| collective-comm | 4/59 (6.8%) | - | - |
| compilation | 15/104 (14.4%) | - | - |
| concurrency-sync | 7/61 (11.5%) | - | - |
| config-observability | 16/78 (20.5%) | - | - |
| host-overhead | 10/85 (11.8%) | - | - |
| inference-serving | 3/44 (6.8%) | - | - |
| io-startup | 2/26 (7.7%) | - | - |
| kernel-efficiency | 96/319 (30.1%) | - | - |
| memory-footprint | 11/41 (26.8%) | - | - |
| memory-management | 7/92 (7.6%) | - | - |
| n/a | 0/1 (0.0%) | - | - |
| parallelism-scheduling | 3/24 (12.5%) | - | - |

## 消融表（§4.2 subset, 1334; detector_v1 同口径 restricted）

| variant | overall recall@2 | regression-fix@2 | benign FPR | Δ recall@2 vs full |
|---|---|---|---|---|
| detector_v1 (all legs) | 174/934 (18.6%) | 32/261 (12.3%) | 56/400 (14.0%) | — |
| ablate_leg1 | 210/934 (22.5%) | 41/261 (15.7%) | 46/400 (11.5%) | +3.9pp |
| ablate_leg2 | 195/934 (20.9%) | 33/261 (12.6%) | 62/400 (15.5%) | +2.2pp |
| ablate_leg3 | 33/934 (3.5%) | 9/261 (3.4%) | 13/400 (3.2%) | -15.1pp |
| ablate_adversarial | 223/934 (23.9%) | 47/261 (18.0%) | 67/400 (16.8%) | +5.2pp |

## 对抗验证前后（论文图，§4.2 subset）

| | recall@2 | benign FPR |
|---|---|---|
| 关闭对抗层 | 223/934 (23.9%) | 67/400 (16.8%) |
| 开启（detector_v1） | 174/934 (18.6%) | 56/400 (14.0%) |

## 成本 / item（full test subset）

| detector | $/item | s/item |
|---|--:|--:|
| **detector_v1** | $0.233 | 18.9 |
| (a) Megatron-strict | $0.347 | 25.5 |
| (b) generic | $0.245 | 18.1 |
| (c) keyword | $0.000 | 0.0 |
| (d) cross-family | $0.000 | 1.7 |

## 诚实小节（必读）

- **北极星结果**：detector_v1 的 regression-fix recall@2 = 32/261 (12.3%)。与两条 Opus baseline（各 14.9%）相比，detector_v1 在**最难的 regression-fix 子类上并未胜出**；其 overall recall@2（174/934 (18.6%)）高于所有 baseline，主要由 optimization 类拉动。这是一个**如实报告的混合结果**：多腿+对抗验证在广度与 FPR 控制上有优势，但对最难子类不优于裸强模型。
- **judge 是 LLM**：命中判定用 Opus（κ=0.926 双判一致，跨家族 Nova 一致 89%），非人工真值。
- **数据集机器标注 + 机器仲裁 + 机器分类**：无人工 ground truth（缓解见 S7 人工审计抽样包）。
- **FPR 基于抽样**：主表 FPR 用完整 954 负样本；消融 FPR 用 400 负样本子样（§4.2），CI 已给出。
- **静态天花板 57.1% 自带标注不确定性**：其定义依赖机器判定的 static_detectability。
- **未做 GPU 复现**：leg3 只产出建议触发的 perf recipe，未实际运行基准。
- leak_attempts 全程 = 0：呈现规则未被违反。

## 消融的关键结论（论文核心诚实点）

消融清楚显示各组件的真实贡献：

- **leg3（风险路由）是唯一净正贡献的腿**：去掉它 overall recall@2 从 18.6% 崩到 **3.5%**
  （−15.1pp），regression-fix 从 12.3% 到 3.4%。它承载了检测器几乎全部的召回。
- **leg1（静态规则）与 leg2（检索 review）在 test 上是净负贡献**：去掉任一，recall 反而
  上升（+3.9pp / +2.2pp）。它们在 dev 上验证时有效，但对 test 的最难子类未能泛化，
  且引入的低质 finding 挤占了 budget=2 的名额。
- **对抗验证层在 test 上净损害**：去掉它 recall@2 +5.2pp（regression-fix +5.7pp 至 18.0%），
  而 benign FPR 仅上升 2.8pp。它在 dev（降 7pp FPR）与 test（降 2.8pp）之间**未能泛化**。

**如实结论**：冻结的 detector_v1（全组件 + 对抗层）在北极星上 12.3%，**低于**其自身的
`ablate_adversarial` 变体（18.0%）与两条 Opus baseline（14.9%）。事后看，最优配置是
"仅 leg3、无对抗层"。但**配置已在 Stage 3 冻结、test 只跑一次**，我们如实上报冻结版结果，
并以消融揭示更优配置的存在——这正是"不用 test 反馈驱动改动"的方法学要求。改进留给 v2。
