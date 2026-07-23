# Paper claims — 每条主张 + 支撑文件/数字 + 已知反驳 (Stage 8 §8.3)

> 规则：每条可写进论文的主张都必须能指向本仓库里的具体文件与数字；每条都附"已知反驳"。
> 数字口径：静态天花板 57.1%，judge = Opus 4.8（κ=0.926，跨家族 Nova 一致 89%），
> test 冻结子集 1888（818 perf-pos + 116 pairs + 954 neg），test 只跑一次，全程 leak=0。

---

## C1. Novelty — 首个 AI-infra 性能回归（引入→修复）配对数据集

**主张**：MegaPerfBench 是**首个** AI 基础设施领域的**性能回归"引入 commit → 修复 commit"
配对数据集**（622 对，A 级 502 / B 级 120），覆盖 Megatron-LM / vLLM / DeepSpeed /
TransformerEngine 四个真实框架。

- **支撑**：`pairs/regression_pairs_v011.jsonl`（622 对）；`reports/dataset_stats.md`；
  `DATASHEET.md`。
- **已知反驳（必须在论文中主动澄清）**：**不可**表述为"首个 AI-infra 性能数据集"——
  该更强表述已被证伪：arXiv **2506.09713**（929-bug 数据集）、**2512.20345**（849 issues）、
  **2506.10426**（308 bugs）均已公开可下载。我们的 novelty 严格限定在
  **"inducing→fix 配对"**这一结构上，而非"首个性能数据集"。

## C2. 任务难度不在 commit 文本里

**主张**：检测性能回归需要理解代码语义，而非读懂 commit message。

- **支撑**：keyword baseline（只看 message + 文件名）在 regression-fix recall@2 = **0.0%**
  （`reports/baseline_results.md`，`paper/tables/T4_per_kind.csv`）；502 个 A 级标签来自
  开发者原话，若任务在文字里，keyword 应该能蹭到分。
- **已知反驳**：keyword 规则集可能不够全 → 缓解：我们用了 9 个高频性能词根且大小写不敏感，
  且它在负样本上确实会触发（FPR 5%），证明它"会报"，只是报不准。

## C3. leaderboard 不只是测"与标注者(Opus)的一致性"

**主张**：跨家族（非 Anthropic）模型独立评测，排名结构与 Opus 系一致，排除自我偏好。

- **支撑**：baseline (d) = Nova Pro（Bedrock Converse）零样本，regression-fix recall@2 = 8.0%、
  benign FPR 23%（`reports/baseline_results.md`）；judge 校准里 Nova 与 Opus-judge 命中一致率
  89%（`reports/judge_calibration.md`）。数据由 Opus 标注仲裁，故此项为必需的自我偏好检查。
- **已知反驳**：Nova ≠ 所有非 Anthropic 家族 → 缓解：judge 校准阶段亦验证了 Llama-3.3-70B 可用；
  报告已按"较低值作为保守 recall"口径处理 >15pp 分歧。

## C4. detector_v1：广度与成本上有优势，但最难子类不优于裸强模型（诚实混合结果）

**主张**：三腿+对抗验证的工程化检测器在 overall recall@2（18.6%）上高于所有 baseline，
单价更低（$0.233/item vs Megatron-strict $0.347），但在**北极星 regression-fix recall@2
（12.3%）上并未超过两条 Opus baseline（Megatron-strict 13.8% / generic 14.2%，持久化 cache 口径）**。

- **支撑**：`reports/detector_v1_results.md`；`paper/tables/T1_baseline_comparison.csv`
  （RUN3 §0.3 从持久化 judge cache 重建，0 miss；口径说明见 `reports/tables_rebuild_note.md`）。
- **已知反驳（我们主动给出）**：见 C5——消融显示这个结果其实被"冻结进去的对抗层 + 两条
  净负腿"拖累；这是**如实上报冻结配置**的结果，非检测器能力上限。

## C5. 消融揭示的组件贡献结构（论文核心诚实发现）

**主张**：leg3（风险路由）承载了检测器几乎全部召回；leg1/leg2/对抗层在 test 上为净负。

- **支撑**：`reports/detector_v1_results.md` 消融表 / `paper/tables/T2_ablation.csv`：
  去 leg3 → overall@2 从 18.6% 崩到 3.5%（−15.1pp）；去 leg1 +3.9pp、去 leg2 +2.2pp、
  去对抗层 +5.2pp（recall 反升）。事后最优 = "仅 leg3、无对抗层"。
- **已知反驳**：这是单次 test 结果，可能过拟合于该子集 → 缓解：dev 上对抗层曾降 7pp FPR，
  test 上只降 2.8pp，说明是**泛化失败**而非噪声；我们不据 test 反馈改配置（方法学要求），
  改进留 v2。

## C6. 对抗验证在 dev↔test 之间泛化失败

**主张**：对抗验证层的收益（降 FPR）在 dev 上显著、在 test 上大幅缩水，而召回代价在两者上都在。

- **支撑**：dev（`reports/detector_v1_dev.md`）对抗前后 25.3%/19% → 21.3%/12%（降 7pp FPR）；
  test（`paper/figures_data/adversarial_before_after.csv`）23.9%/16.8% → 18.6%/14.0%（仅降 2.8pp FPR）。

## C7. 检测器侧相关工作与弱 baseline 参照

**主张**：与提交级性能回归风险预测的既有工作相比，本工作提供了配对数据 + 多口径评测。

- **支撑/必引**：arXiv **2604.00222**（2026-04，commit 级性能回归风险预测，ROC-AUC 0.694）——
  它既是相关工作，也是一个弱 baseline 参照点（我们的任务是 finding 级命中，不是二分类风险打分，
  口径不同，需在论文中说明不可直接比较）。

## C8. Tier-A 归因（开发者原话）的证据强度主张

**主张**：本数据 502 个 A 级 inducing→fix 归因来自开发者原话，其证据强度可主张**高于算法 SZZ**。

- **支撑/必引**：ICSE 2021 的 **developer-informed oracle** 正是同一口径，而最优 SZZ 变体对其
  仅 **61% F-measure**——即"开发者原话"作为 oracle 显著强于纯算法归因。见 `pairs/` 的
  `evidence_tier=A` / `evidence_source=commit-message`。
- **已知反驳**：开发者原话也可能错/不完整 → 缓解：B 级 120 对用更弱证据单列；S7 人工审计包
  （`audit/`）留出人工复核 A/B 两组的空间。

## C9. 无记忆污染，主指标可在全 test 集上报

**主张**：裸模型无法从记忆复现这些 commit 的性能后果，test 无记忆污染。

- **支撑**：`reports/memorization.md`——全量 934（818 pos + 116 pairs）探测，**0 memorized**。

---

## 汇总数字表（供正文引用）

| 主张 | 关键数字 | 文件 |
|---|---|---|
| C2 keyword | regfix@2 = 0.0% | T4_per_kind.csv |
| C3 跨家族 | Nova regfix@2 8.0% / FPR 23% / judge一致89% | baseline_results.md, judge_calibration.md |
| C4 detector | overall@2 18.6%, regfix@2 12.3%, $0.233/item | T1_baseline_comparison.csv |
| C5 消融 | 去leg3 −15.1pp；去对抗 +5.2pp | T2_ablation.csv |
| C9 memorization | 0/934 | memorization.md |
| 天花板 | 57.1% | 各 recall 表置顶 |
