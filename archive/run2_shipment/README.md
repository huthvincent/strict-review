# MegaPerfBench Run2 — 检测器线交付包

> 生成日期：2026-07-19 · 对应 git tag `run2-complete`
> 依据指令：`RUN2_INSTRUCTIONS.md`（随包附上）
> 工作目录（正本）：`/mnt/efs/tsfm/rui/GAA/ai_infra_run2/ai_infra_v011_work/`
> 模型：所有 Anthropic 调用 = `us.anthropic.claude-opus-4-8`；跨家族 = Nova Pro / Llama-3.3-70B（Bedrock Converse）

本包是 Run2「检测器线」的**完整交付**：一个可用的性能回归检测器 + 一份完整成绩单 +
一篇论文所需的全部实验 + 待人工判定的抽样包。**8 个 Stage 全部完成，全程 `leak_attempt = 0`。**

---

## 0. 一分钟结论（TL;DR）

| 指标（test 冻结子集 1888 = 818 perf-pos + 116 pairs + 954 neg） | detector_v1 | 最强 baseline |
|---|---|---|
| **regression-fix recall@2**（北极星，n≈145） | **12.3%** | 14.9%（Megatron-strict / generic） |
| overall recall@2 | **18.6%**（高于所有 baseline） | 13.1%（Megatron-strict） |
| benign FPR（加权，Wilson CI） | 14.9% [12.8,17.3] | — |
| 成本 | **$0.233/item**（比对手便宜） | $0.347（Megatron-strict） |
| 静态天花板 | 57.1% | — |

**核心诚实结论（消融得出，务必看 §4）**：检测器的召回几乎全部来自 **leg3（风险路由）**；
**leg1/leg2/对抗验证层在 test 上是净负贡献**。冻结的 detector_v1（含对抗层，regfix 12.3%）
被它自己的「关掉对抗层」变体（**18.0%**）反超。这是**如实上报冻结配置**的结果——按方法学
纪律，配置在 Stage 3 冻结、test 只跑一次、不用 test 反馈改配置。事后最优 =「仅 leg3、
重设计对抗层」，留给 v2。

---

## 1. 两条不可违反的方法学纪律（本包全程遵守）

1. **test 只在 Stage 4 碰一次。** 所有构建、调参、prompt 迭代、阈值选择只用 `dev`。
2. **呈现规则（protocol.v1 §2）**：检测器只看 inducing-commit 时点的 `(diff, message, parent 树快照)`；
   card 的后验字段（mechanism/evidence/magnitude/issue_refs/pr_ref/taxonomy_label/fix commit）
   **绝不进入检测器输入**。harness 从 git 重建视图，永不把 card 递给检测器。
   → **验证：全部预测文件的 `leak_attempt` 累计 = 0**（见 `predictions/`，每行都带该字段）。

---

## 2. 目录导览

```
README.md                     ← 本文件（总说明）
RUN2_INSTRUCTIONS.md          ← 本轮唯一指令书（自包含，供追溯）
report.md                     ← 项目进度总览（含 Run2 banner）
DATASHEET.md                  ← 数据集 datasheet（含机器标注诚实声明）

reports/                      ← 所有阶段报告（每个 Stage 的成绩单）
  judge_calibration.md          S1: judge κ=0.926 + 跨家族校验
  memorization.md               S1: 全量 934 探测 / 0 memorized
  baseline_results.md           S2: 四 baseline × 全指标 + GATE
  leg1_rules.md                 S3: 静态规则挖掘（31 留存 / FPR 2.6%）
  detector_v1_dev.md            S3: dev 调参 + 对抗验证前后
  detector_v1_results.md        S4: ★ test 主对决 + 消融 + 诚实小节
  split_repo_holdout.md         S5: 跨仓泛化（迁移损失 +4.8pp）
  prospective_run.md            S6: ★ 61 未见新 commit 前瞻 + top-20 人工核验清单
  leg1_rules_noMega.md          S5: 排除 Megatron 的规则集（22 留存）
  human_audit_packet.md         S7: 抽样方法 + agreement/CI 计算口径

detector/                     ← 冻结的检测器 v1（可复现）
  detector_v1.config.json       ★ 冻结超参（tag detector-v1-frozen 后不可改）
  detector_v1_noMega.config.json  S5 跨仓变体配置
  leg3_recipe_map.json          ★ 叶→Megatron perf-CI recipe 映射（可交付 NVIDIA）
  ruleset_kept.jsonl            leg1 留存的 31 条可执行静态规则
  ruleset_kept_noMega.jsonl     S5 排除 Megatron 的 22 条规则
  ruleset_rejected.jsonl        被拒规则 + 理由（负结果也是论文材料）
  leg2_kb_note.txt              leg2 知识库说明（KB 本体见工作目录，3386 条，可再生）

predictions/                  ← 全部原始预测（证据，每行含 leak_attempt / cost / findings）
  <detector>/<split>.jsonl      detector_v1 / 4 baseline / 4 ablation / noMega / prospective

paper/                        ← Stage 8 论文素材
  tables/*.md, *.csv            T1 baseline对比 / T2 消融 / T3 taxonomy / T4 per-kind / T5 FPR
  figures_data/*.csv            漏斗/taxonomy/detectability/对抗前后/recall@budget 曲线
  claims.md                     ★ 9 条主张 + 支撑文件数字 + 已知反驳（含三处必引 arXiv）
  limitations.md                9 条 limitation + 缓解

audit/                        ← Stage 7 人工审计抽样包（只备材料，不产标签）
  sample_manifest.json          抽样方法 + seed + 分层计数
  human_audit_packet.md         360 卡 + 150 对，可离线直读（含 diff hunk + 待填空位）
  human_audit_template.jsonl    预填 machine_value，verdict 留空，可被 apply_audits 消费

splits/                       ← 冻结评测集 + 探测标注
  test_eval_subset.txt          ★ 冻结 test 子集（1888，所有参赛者跑同一集）
  TEST_SUBSET_FROZEN            冻结哨兵 + 去重说明
  test_ablation_subset.txt      §4.2 消融子集（1334 = pos全量 + 400 neg子样）
  test_megatron.txt             S5 用的 Megatron test 部分（280）
  dev_calib.txt / dev_tune.txt  S1 校准 / S3 调参用子集
  memorization_flags.jsonl      934 条记忆探测原始输出

scripts/                      ← 全部可复现代码
  eval_harness.py               harness.v1：PR-time 视图 + parent 锚定工具 + leak-guard
  test_harness_leakage.py       泄漏自检（3/3 PASS）
  eval_metrics.py               metrics.v1：judge + recall@budget + FPR + κ + 磁盘缓存
  freeze_test_subset.py         冻结 test 子集
  detectors_baseline.py         四 baseline（含 prompt-caching）
  leg1_mine_rules.py            leg1 规则挖掘
  run_rules.py                  leg1 规则运行 + dev 验证
  leg2_retrieval.py             leg2 BM25 检索增强 review
  leg3_routing.py               leg3 风险路由 + recipe 映射
  detector_v1.py                ★ 融合 + 对抗验证层（含消融支持）
  detector_dev_report.py        S3 dev 报告
  ablation_report.py            S4 消融报告
  build_audit_packet.py         S7 抽样包
  build_paper_materials.py      S8 论文素材
  prospective_run.py            S6 前瞻试运行

prospective/
  unseen_commits.json           61 个数据集未见的 Megatron 新 commit（git-fetch 得，含窗口说明）
```

★ = 论文/交付最关键的产物。

---

## 3. 各 Stage 结果速览

### S1 评测地基（tag `s1-complete`）
- harness.v1 + 泄漏自检 **3/3 PASS**；冻结子集 **1888**（修正了两个数据 bug：389 个非-perf
  case 混入 + 41 个重复 hard-negative-hotfile 行，均见 `splits/TEST_SUBSET_FROZEN`）。
- **judge 校准 GATE 通过：Cohen's κ = 0.926**，双判一致 97%，跨家族（Nova）一致 89%。
- **memorization：全量 934（818 pos + 116 pairs）探测，0 memorized** → novel 子集 = 全集。

### S2 四 baseline（tag `baseline-v1`，GATE 通过）
| baseline | regression-fix@2 | benign FPR | 意义 |
|---|---|---|---|
| (a) Megatron-strict（对手，Opus） | 14.9% | 17% | 最强模型也只有 14.9% → 任务有挑战性，GATE ∈[5,45%] ✓ |
| (b) generic（裸 Opus） | 14.9% | 5% | FPR 最低 |
| (c) keyword（无 LLM） | **0.0%** | 5% | ★ 证明任务难度不在 commit 文字里 |
| (d) cross-family（Nova） | 8.0% | 23% | ★ 排除「只测与标注者一致性」的自我偏好 |

### S3 检测器 v1（tag `detector-v1-frozen`）
- 三腿：leg1 静态规则（31 条留存 / dev FPR 2.6%）· leg2 BM25 检索增强 · leg3 风险路由。
- 对抗验证层（只作用 leg2）+ budget-2。dev 调参 recall@2 **21.3%**。
- 冻结依据与配置全部记在 `detector/detector_v1.config.json` 的 `_note`。

### S4 test 对决（tag `detector-v1`，test 唯一一次）
主表（完整 1888 子集）+ **消融表**（§4.2 的 1334 子集）：

| 变体 | overall recall@2 | regression-fix@2 | benign FPR | 去掉后 Δrecall |
|---|---|---|---|---|
| **detector_v1（全腿+对抗）** | 18.6% | 12.3% | 14.0% | — |
| 去 leg1 | 22.5% | 15.7% | 11.5% | **+3.9pp** |
| 去 leg2 | 20.9% | 12.6% | 15.5% | +2.2pp |
| **去 leg3** | **3.5%** | **3.4%** | 3.2% | **−15.1pp** |
| 去对抗层 | 23.9% | 18.0% | 16.8% | **+5.2pp** |

→ **leg3 承载几乎全部召回；leg1/leg2/对抗层在 test 上净负。** 详见 `reports/detector_v1_results.md`。

### S5 跨仓泛化
排除 Megatron 重建 leg1+leg2 → 在 Megatron test 部分对比：**迁移损失 +4.8pp overall@2**
（全量知识 12.4% vs 排除 7.6%；regression-fix 两版都 8.0%）。

### S6 前瞻试运行（真实新 commit）
61 个数据集**未见**的 Megatron 新 commit（git-fetch，窗口 2026-06-15..07-19），
**43% 触发率，leak=0**。top findings 命中正确 taxonomy——例如
`#5834 "Avoid extra MFSDP v2 model-weight sync memcpy"` → 检测器独立报
`redundant-buffer-copy`（subject 自证机制）。见 `reports/prospective_run.md` 的 top-20 核验清单。

### S7 人工审计包
A 组 360 卡（全 11 类，每类 ≥5）+ B 组 150 对（A 级 100 / B 级 50），确定性 seed。
**只备材料不产标签**，建议优先判 B 组。

### S8 论文素材
5 张主表 + 5 组作图数据 + `claims.md`（9 主张，含必引 arXiv 2506.09713 / 2512.20345 /
2506.10426 / 2604.00222 与 ICSE21 developer-informed oracle）+ `limitations.md`（9 条含缓解）。

---

## 4. 如何复现

环境（见 `RUN2_INSTRUCTIONS.md` 与工作目录内 skill_opt/API.md）：
```bash
cd /mnt/efs/tsfm/rui/skill_opt
export MPB_BASE=<工作目录> MPB_REPOS=/home/ec2-user/megaperf_repos AWS_REGION=us-east-1
~/.local/bin/uv run --no-project python <脚本>
```
- 重跑 detector_v1 的 test：`scripts/detector_v1.py --split test_eval_subset`
- 重出主表：`scripts/ablation_report.py`（judge 结果有磁盘缓存，重跑近乎免费）
- 重跑某条腿的消融：`scripts/detector_v1.py --split test_ablation_subset --ablate <leg1|leg2|leg3|adversarial>`

注：`scripts/` 依赖工作目录内的 `mpb_common.py`、数据集（cases/pairs/negatives/taxonomy）与
四个仓库 clone（`$MPB_REPOS`）；这些是 v0.2 **数据集输入**，体量大（~90MB+clone），
**未含在本结果包内**，在正本工作目录中。本包聚焦 Run2 检测器线的产物、证据、代码与说明。

---

## 4b. 过程中的修正与基础设施加固（透明记录）

**修的两个数据 bug（都在 test 未被任何检测器触碰之前，故安全）：**
1. **冻结子集混入 389 个非-perf case**：原 `freeze_test_subset.py` 取了 test.txt 全部 `case:`
   行，其中 389 个 `is_perf_related=false` 既非正样本也非负样本，会被跑却不计分。已过滤，
   子集正样本严格 = 818 个 perf-related case（符合 spec §1.3）。
2. **冻结子集含 41 个重复 hard-negative-hotfile 行**：源自 v0.2 数据 test.txt 的既有重复
   （`negatives_v011.jsonl` 有 10076 个重复 case_id）。已就地去重 → 唯一负样本 954（非目标
   995，hotfile 因此 459/500，已在 `splits/TEST_SUBSET_FROZEN` 记录）。四条 baseline 的预测
   文件同步去重，**零重算浪费**。freeze 脚本现对 pos/pairs/neg 池全部去重。

**两项基础设施加固（保证评测可复现、抗瞬时故障）：**
1. **judge 磁盘缓存**（`eval_metrics.py`）：judge 判定按「prompt 精确哈希」持久化到
   `.judge_cache.jsonl`，相同 (ground-truth, findings) 不重判 → 重出主表/消融近乎免费，
   且保证同一预测的评分完全确定性可复现。
2. **judge 瞬时错误重试 + 逐项 fail-soft**（`eval_metrics._opus_judge`）：Bedrock 流式
   调用偶发 `Internal server error` / `ReadTimeout` 曾多次整段打断打分 pass。加固为
   6 次指数退避重试；仍失败则该单项记为 non-hit 并继续，**不再让一次网络抖动作废整轮评测**。
   （代价：极少数条目可能被判为 non-hit，方向保守，不会虚高 recall。）

**成本超支说明**：见下方诚实声明第 7 条。

---

## 5. 诚实声明（必读，与 DATASHEET 配套）

1. **数据集全机器标注 + 机器仲裁 + 机器分类，无人工 ground truth**（S7 抽样包用于后续人工复核）。
2. **judge 是 LLM**（Opus，κ=0.926 + 跨家族 89%），非人类真值。
3. **检测器 v1 冻结配置非最优**：消融显示事后最优 =「仅 leg3、重设计对抗层」；如实上报冻结版。
4. **FPR 基于抽样**：主表用全量 954 负样本；消融用 400 负样本子样（§4.2），均附 Wilson CI。
5. **静态天花板 57.1% 自带标注不确定性**（依赖机器判定的 static_detectability）。
6. **未做 GPU 复现**：leg3 只产出建议触发的 perf recipe，未实际跑基准。
7. **成本**：Run2 总花费 ~$3,167（Rui 批准 ~$3k，略超 $167，主因是两条 Opus baseline 跑全量
   1888 子集 + 全保真消融矩阵）。spec 原估 ~$1,450 / 硬帽 $2k 已由 Rui 上调至 ~$3k。

**下一步（v2 方向、投稿、与 NVIDIA 接触）由 Rui 与规划方决定。**
