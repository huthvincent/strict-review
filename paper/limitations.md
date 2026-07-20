# Limitations & mitigations (Stage 8 §8.4)

每条 limitation 附缓解措施与未来工作。与 `DATASHEET.md` 的诚实声明配套。

## L1. 无人工 ground truth（最大弱点）

数据集**全机器标注 + 机器仲裁 + 机器分类**。同域前作有人工真值（如 308/849-bug 数据集为
100% 人工标注、κ 0.84–0.90），本数据集没有。

- **缓解**：(a) Tier-A 归因来自开发者原话（commit message），非算法推断；(b) S7 已备好
  人工审计抽样包（`audit/`：360 卡 + 150 对，全 11 类覆盖，可回填模板），把人工成本压到最低；
  (c) 仲裁 κ 与跨家族一致率均已报告。
- **未来工作**：执行 S7 抽样包的人工判定，报 machine-vs-human agreement + κ + Wilson CI。

## L2. judge 是 LLM，非人类

命中判定用 Opus 4.8。虽 κ=0.926（双判一致 97%）、跨家族 Nova 一致 89%，但仍是模型判模型。

- **缓解**：双判 + 跨家族交叉校验；>15pp 分歧时以较低值作保守 recall。
- **未来工作**：对 recall@2 命中样本做人工抽检，校准 judge 偏差。

## L3. 检测器侧对手模型偏差

baseline (a) Megatron-strict 用 Opus 4.8 + NVIDIA 实际内部模型不公开，故给对手配了最强可得模型。
这**高估**了"NVIDIA 现状"这一对照的强度。

- **缓解**：已在 `reports/baseline_results.md` 明确记录此偏差；同时给出裸 generic 与跨家族对照。

## L4. FPR 基于抽样

主表 FPR 用完整 954 负样本；**消融** FPR 用 400 负样本子样（§4.2 授权的降本口径）。

- **缓解**：所有 FPR 均附 Wilson 95% CI；主表用全量负样本。
- **未来工作**：预算允许时消融也跑全量负样本。

## L5. 静态天花板 57.1% 自带标注不确定性

天花板 = test 正样本中 `static_detectability=low` 的占比（351/818）。该字段由机器判定，
故天花板本身继承标注不确定性。

- **缓解**：天花板作为"参照线"而非硬上界呈现；每张 recall 表置顶标注。

## L6. 未做 GPU 复现

leg3 只产出"建议触发的 Megatron perf recipe"，未实际跑基准验证回归量级；magnitude 覆盖仅 6%。

- **缓解**：leg3 的 recipe 映射（`detectors/leg3_recipe_map.json`）是可交付 NVIDIA 的产物，
  设计上就是把验证外包给已有 perf CI，而非自己跑 GPU。
- **未来工作**：在真实 GPU CI 上执行 top 建议，闭合"找出→验证"环。

## L7. 检测器 v1 冻结配置非最优

消融显示事后最优 = "仅 leg3、无对抗层"，但冻结的 detector_v1 含全部三腿 + 对抗层。

- **缓解**：这是**方法学纪律**的代价——配置在 Stage 3 冻结、test 只碰一次，不能用 test 反馈
  改配置。我们如实上报冻结版并用消融揭示更优配置的存在。
- **未来工作**：v2 用 dev 上的组件级消融来选腿（而非整体调参），并重新设计对抗层的泛化。

## L8. 单仓库主导 + 跨仓泛化损失

vLLM 占 perf 卡的多数（3300/5016）；跨仓泛化实验（S5）量化了"知识库只用非 Megatron、测 Megatron"
的迁移损失。

- **缓解**：S5 报告（`reports/split_repo_holdout.md`）给出 detector_v1 vs detector_v1_noMega
  在 Megatron test 部分的 recall@2 差值。
- **未来工作**：按仓库分层重平衡知识库。

## L9. 时间泄漏 / 数据新鲜度

数据集截止某快照；检测器在历史 commit 上评测。为回答"能否上线用",S6 在**数据集完全没见过的**
2026-06-15 之后的 61 个 Megatron 新 commit 上做了前瞻试运行（`reports/prospective_run.md`）。

- **缓解**：S6 定性产出 + top-20 人工核验清单;呈现规则保证检测器只看 inducing-commit 时点视图。
