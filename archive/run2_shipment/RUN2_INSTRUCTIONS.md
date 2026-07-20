# 执行机作业书 · 第二轮（检测器线全量）

> **这是本轮唯一的指令文件**，自包含。旧的 `DETECTOR_LINE_INSTRUCTIONS.md` /
> `DATASET_COMPLETION_INSTRUCTIONS.md` / `S2_INSTRUCTIONS.md` 已归档到
> `archive/instructions/`，仅供追溯，不再执行。
>
> **本轮目标**：一次跑完检测器线的全部工作 —— 建评测地基 → 拿到"要打败的数字" →
> 造检测器 v1 → 冻结对决 → 泛化实验 → 在真实新 commit 上试运行 → 备好论文素材。
> 跑完后 Rui 手上应当有：**一个可用的检测器 + 一份完整成绩单 + 一篇论文所需的全部实验**。
>
> 模型纪律：默认 `us.anthropic.claude-opus-4-8`（Bedrock 标准通道）。
> **例外**：Stage 2d 与 Stage 1 的交叉校验必须用**非 Anthropic** 模型（见各处说明）。
> 预算硬帽 **$2,000**（分帽见各 Stage；估算合计 ~$1,450）。
> 存储纪律不变：只写本目录、本地 git 无 remote。除 Stage 6 的 `git fetch` 外无外网需求。

---

## 两条不可违反的方法学纪律

1. **test 只在 Stage 4 碰一次。** Stage 3 的一切构建、调参、prompt 迭代、阈值选择
   只准用 `dev`。在 test 上"看一眼再改"会让最终数字失去意义。
2. **呈现规则（protocol.v1 §2）是硬约束。** 检测器只能看到 inducing commit 时点的
   `(diff, message, parent 树快照)`。card 的后验字段（`mechanism`、`evidence`、
   `magnitude_reported`、`issue_refs`、`pr_ref`、`taxonomy_label`、fix commit）
   **绝不可**进入检测器输入。harness 必须从 git 重建视图，永不把 card 递给检测器。
   违反即评测作废。

---

## 已知地基（直接用，不必重推）

| 项 | 值 |
|---|---|
| test 正样本 | **818**（regression-fix **145** / optimization 615 / config-default-change 38 / 其他 20） |
| test 负样本 | **3,120**（hard-neg-hotfile 2,237 · random-benign 621 · false-signal-smoke-ci 217 · false-signal-perf-infra 42 · lookalike 3） |
| test 回归对 | **116** |
| dev（构建调参） | 814 正 / 4,243 负 / 119 pairs |
| train（知识库/规则挖掘） | 3,412 正 / 22,390 负 / 387 pairs |
| **静态天花板** | **57.1%**（test 正样本 `static_detectability=low` 占 351/818） |
| memorization | 已探 196 项，**0 memorized** |
| 分类学 | 74 叶 / 11 大类；卡片 `taxonomy_label` 存**叶子 id** |
| split 行格式 | `case:{case_id}` / `neg:{case_id}` / `pair:{pair_id}` |
| clone | `$MPB_REPOS`（默认 `/home/ec2-user/megaperf_repos`） |

已备好、直接可用的 prompt（**不要自己重写**）：
`prompts/baseline_megatron.v1.md`（对手，钉版 `a3d761b28f77` 逐字忠实适配）·
`prompts/baseline_generic.v1.md`（裸模型对照）·
`prompts/eval_judge.v1.md`（命中裁判，含四个正反例）。

---

# Stage 1 — 评测地基（$60）★ 硬 gate

## 1.1 harness（`scripts/eval_harness.py`，`harness.v1`）

把一个 item 变成 PR-time 视图，递给任意检测器，收回结构化 findings。

- case item 目标 sha = 该 case 的 sha；pair item 目标 sha = **`inducing_sha`**
- 视图 = `{repo, sha, parent_sha, author_date, commit_message, diff, changed_files}`；
  `diff` = `git show <sha>`（合并提交取首父），裁剪复用 `diff_packer.py` 的 `packer.v1`
  （非测试 hunk 优先，≤8k tokens，记 `diff_truncated`）
- 检测器可用工具，全部锚定 **parent 快照**：`read_file_at_parent(path)` ·
  `grep_at_parent(pattern, glob)` · `git_log_before(path, n)`。任何解析到目标 sha
  之后的引用必须拒绝并置 `leak_attempt: true`
- 统一接口与落盘：
  ```python
  def detect(view: dict, tools: Tools) -> list[Finding]
  # Finding = {severity: "critical"|"important"|"suggestion", category: str,
  #            file: str|None, line: int|None, claim: str,
  #            confidence: float, suggested_benchmark: str|None}
  ```
  → `predictions/{detector_id}/{split}.jsonl`：
  `{item_id, findings, n_turns, tokens, cost, latency_s, leak_attempt}`

**泄漏自检（必须写成测试 `scripts/test_harness_leakage.py`，不过不许继续）**：
(a) 视图键集合 ⊆ 白名单；(b) 随机 50 个 item 的视图 JSON 不含该 case 的
`mechanism`/`evidence`/`taxonomy_label`/`magnitude_reported` 的任何 >20 字符子串；
(c) 工具越界被拒。

## 1.2 指标（`scripts/eval_metrics.py`，`metrics.v1`）

按 protocol.v1 §4 实现：

- **命中判定**：用 `prompts/eval_judge.v1.md`，输入 = ground truth（mechanism +
  evidence + symptom）+ 检测器 findings，输出 `{hit, hit_finding_index, rationale}`
- **judge 校准（必做）**：dev 上抽 200 对做双 judge 独立判定，报一致率与 κ
  → `reports/judge_calibration.md`；**κ < 0.7 先修 prompt 再继续**
- **跨家族交叉校验（必做）**：上述 200 对中抽 100 对，用**非 Anthropic 模型**
  （Bedrock 上 Nova Pro / Llama 3.x，以可用者为准）同 prompt 独立判定，报与
  Opus-judge 的一致率。差异 > 15pp 则显著标注，并以**较低值**作为保守 recall 上报
- **recall@budget**：findings 按 confidence 降序取前 budget 条（**只计 severity ∈
  {critical, important}**），budget ∈ {1,2,5}，**budget=2 为北极星**
- **per-kind recall**：三类分开；**北极星 = regression-fix 的 recall@2（n=145）**
- **per-taxonomy recall**：按 11 大类聚合
- **benign FPR**：负样本上出现 severity ∈ {critical, important} 且 confidence ≥ 0.5
  即计 FP；**按 negative_type 分层** + 加权总值 + **Wilson 95% CI**
- **inducing 归因准确率**（pairs，可选实现）：按 evidence_tier A/B 分开
- **静态天花板**：每张 recall 表顶部固定打印 `ceiling = 57.1%`

## 1.3 冻结 test 评测子集（`splits/test_eval_subset.txt`）

正样本 818 全用 · pairs 116 全用 · 负样本确定性分层抽样（`random.Random(20260718)`）：
hard-neg-hotfile **500**/2237 · random-benign **300**/621 · false-signal-smoke-ci
**150**/217 · false-signal-perf-infra **42**/42 · lookalike **3**/3 = **995**。
写入后追加哨兵 `splits/TEST_SUBSET_FROZEN`。**所有参赛者必须跑在完全相同的子集上。**

## 1.4 Memorization 补探（$15）

对子集的全部 818 个正样本补跑 `memorization_probe.v1`（裸 Opus，无工具），
追加 `splits/memorization_flags.jsonl`，更新 `reports/memorization.md`。
若仍为 0，报告中写明"novel 子集 = 全集"。

---

# Stage 2 — 四个 Baseline（$420）★ 拿到"要打败的数字"

统一条件：输入是 1.1 的视图、工具是那三个 parent 锚定工具、输出走统一 Finding
schema（`tool_choice` 强制）、`max_turns=6`。全部跑在 1.3 冻结子集上。

| # | Baseline | 模型 | 说明 |
|---|---|---|---|
| a | **Megatron strict-review**（对手） | Opus 4.8 | 用 `prompts/baseline_megatron.v1.md`。NVIDIA 实际模型不公开，给对手配最强模型；此偏差写入报告 |
| b | 裸模型泛型 review | Opus 4.8 | 用 `prompts/baseline_generic.v1.md` |
| c | **commit-message 关键词** | 无 LLM | 只看 message + 文件名，命中 perf/slow/regression/throughput/latency/memory leak/OOM/overhead/optimize 即报一条 `important`。$0 |
| d | **非 Anthropic 家族零样本** | Nova Pro / Llama 等 | 同 (b) 的 prompt 与工具，换模型家族 |

**(c) 和 (d) 是方法学必需项，不可省**：
- (c) 回答"这 benchmark 是不是只在测能不能读懂 commit message"——数据集 502 个 A 级
  标签来自开发者原话，审稿人必问。它的分数越低，越证明任务难度不在文字里。
- (d) 回答"leaderboard 测的是不是与标注者（Opus）的一致性"——数据由 Opus 标注仲裁，
  席上全是 Opus 会被指自我偏好。

**产出** `reports/baseline_results.md`：四条 baseline × 全部指标；天花板置顶；分层 FPR + CI；
per-kind（**regression-fix@2 加粗**）；per-taxonomy；成本/延迟；leak_attempt（应为 0）。
git tag `baseline-v1`。

**GATE**：baseline (a) 的 regression-fix recall@2 若落在 5–45% → 正常，继续 Stage 3。
< 5% 疑似 harness/judge bug → **停**，写 BLOCKERS.md。> 45% → 继续，但在报告中提示
Rui 增量空间被压缩。

---

# Stage 3 — 检测器 v1 三条腿（$550）★ 只用 dev

设计依据：test 正样本按静态可判定性天然三分（high 55 / medium 369 / low 351）。

## 3.1 腿1 · 静态规则生成（主攻 high；论文贡献点）

从历史 patch 自动合成**可独立执行、不调 LLM** 的静态规则。

1. 挖掘集：train 中 `static_detectability=high` 且 kind ∈ {regression-fix, optimization}
   的卡（约 300–400 张），取其 **fix commit 的 diff**（训练侧允许看 fix；test 侧永不允许）
2. Opus 逐张产出规则草案 `{id, 反模式描述, 匹配逻辑, 正例, 反例, 归属叶}`；
   匹配逻辑优先级：Semgrep（若可装）> Python `ast` 检查器 > 受限正则
3. 同叶内语义重复的规则合并
4. **dev 全量验证**：`precision = 命中正样本 /(命中正+命中负)`；
   **只保留 precision ≥ 0.5 且命中 ≥ 2 个 dev 正样本**的规则，其余进
   `rules/rejected.jsonl`（附理由——负结果也是论文材料）
5. 产出 `rules/ruleset.v1/` + `scripts/run_rules.py` + `reports/leg1_rules.md`
6. 成功判据：留存 ≥ 15 条且 dev FPR ≤ 5%。达不到如实记录，**不放宽门槛硬凑**

## 3.2 腿2 · 检索增强 review（主攻 medium；预期主力）

1. 知识库 = train 的 3,412 张 perf 卡，每张压成
   `{case_id, taxonomy_leaf, 一句话机制, 触及 API/符号, 文件类型, symptom}`
2. 检索：BM25/TF-IDF over（改动文件路径 + diff 中符号名 + API 调用）；
   可选叠加 Bedrock Titan embedding 做 A/B
3. 判定：Opus 收 PR-time 视图 + top-k 历史案例（**只含机制与叶子标签，不含本 item
   的任何标签**）+ 命中叶的 `boundary_notes` → findings
4. dev 调参：k、taxonomy 提示、few-shot 数、confidence 阈值 → `reports/leg2_tuning.md`
5. 成功判据：dev 的 medium 子集 recall@2 相对 baseline (b) 提升 ≥ 30%

## 3.3 腿3 · 风险路由（主攻 low；**终极目标的关键**）

对静态看不出的，不硬猜 bug，而是输出可行动的风险信号 —— 这是"找出来并**验证**"
闭环的前半段：我们不自己跑 GPU，而是精准触发 Megatron **已有的** perf CI。

- 输出：`severity: important` 的 finding，`claim` = "此改动触及 {taxonomy 叶}，
  该类问题历史上只在 {manifest 条件} 下显现"；
  `suggested_benchmark` = 建议触发的 recipe（`gpt-perf` / `moe_perf` / `hybrid-perf` /
  `determinism-perf` 之一）+ 建议配置
- 依据：train 中同叶案例的 `manifest_conditions`（S2 回填的真实用户配置在此发挥价值）
- 映射表落盘 `detectors/leg3_recipe_map.json`（叶 → recipe + 触发条件），
  这张表本身就是可交付给 NVIDIA 的产物
- 评测口径：该腿的 finding 参与 recall@budget；judge 认定机制/风险与 ground truth
  吻合才算 HIT，泛泛"建议跑性能测试"判 MISS
- 产出 `reports/leg3_routing.md`：recipe 分布 + dev 的 low 子集 recall@2 + 该腿单独 FPR

## 3.4 融合与对抗验证（`detector_v1`）

1. 路由：轻量分类判断场景（high/medium/low），决定调用顺序；腿1 规则永远先跑（最便宜）
2. 合并去重：同 (file, 机制) 取最高 confidence
3. **对抗验证层**：第二个 Opus 实例扮演反驳者（`prompts/adversarial_verify.v1.md`，
   需新建），任务是论证该 finding 为什么错/无关紧要，输出 `{refuted, reason}`；
   被反驳的降级（critical→important→丢弃）。这是控 FPR 的主要手段
4. 预算裁剪：按 confidence 取前 2 条
5. `reports/detector_v1_dev.md`：含**对抗验证前后的 recall/FPR 对比**（论文的一张图）
6. 冻结 `detectors/detector_v1.config.json`（全部超参/prompt 版本/规则集版本/k/阈值），
   git tag `detector-v1-frozen`，此后不得再改

---

# Stage 4 — test 对决（$280）★ test 唯一一次

前置断言：配置已冻结；本 Stage 期间禁止修改任何 prompt/阈值/规则。效果不佳就如实报告，
改进留 v2。

1. `detector_v1` 在 `splits/test_eval_subset.txt` 上跑完
2. **消融**（同一子集；为省钱可只跑正样本全量 + 负样本 400 条子样，报告标注）：
   `ablate-leg1` · `ablate-leg2` · `ablate-leg3` · **`ablate-adversarial`**（最能说明
   FPR 控制价值）
3. 产出 `reports/detector_v1_results.md`：
   - 置顶：静态天花板 57.1% + memorization 状态
   - 主表：detector_v1 vs baseline (a)(b)(c)(d) × {recall@1/2/5, per-kind
     （**regression-fix@2 加粗**）, per-taxonomy, 分层 FPR + CI, 成本/item}
   - 消融表 + 对抗验证前后对比
   - **诚实小节**：judge 是 LLM（κ + 跨家族一致率）、数据集机器标注、FPR 基于抽样、
     天花板自带标注不确定性、未做 GPU 复现
4. git tag `detector-v1`

---

# Stage 5 — 跨仓库泛化实验（$120）

protocol.v1 §1 规定的 secondary split，论文必需的泛化证据：
**知识库与规则只用 vllm/DeepSpeed/TransformerEngine，测 Megatron-LM。**

1. 重建腿2 知识库与腿1 规则集，**排除全部 Megatron 来源**（记为 `detector_v1_noMega`）
2. 在 test 子集的 **Megatron 部分**跑，与 `detector_v1`（全量知识）在同一子集上对比
3. 产出 `reports/split_repo_holdout.md`：两者 recall@2 差值即"跨仓迁移损失"。
   这直接回答"这套方法只对见过的仓库有效吗"

---

# Stage 6 — 真实新 commit 前瞻试运行（$40）★ 回答"到底能不能用"

历史评测再漂亮，也不等于"上线能用"。本 Stage 在**数据集完全没见过的新提交**上跑一遍。

1. `git fetch` 更新四个 clone（这是本轮唯一需要外网的动作；失败则跳过本 Stage 并记录）
2. 取 **Megatron-LM 在 2026-07-01 之后**的全部新 commit（若不足 80 个，放宽到 06-15 之后；
   记录实际窗口与数量）
3. 用**冻结的** `detector_v1` 逐个跑（同一 harness、同一预算=2）
4. 无标签，故做**定性产出** `reports/prospective_run.md`：
   - 每条 finding 一个区块：commit、finding 原文、confidence、触发的腿、
     `suggested_benchmark`（若有）
   - 按 confidence 降序排列，**top 20 单独成节**供 Rui 人工核验
   - 统计：触发率（多少 commit 被报了问题）、腿的分布、taxonomy 分布
   - **与该窗口内 claude[bot] 的实际评论对比**（若 clone 的 PR 元数据可得则做，
     不可得则跳过并记录）
5. 这一节的价值：若 top 20 里有真问题，那是"这套东西能用"的最直接证据，
   也是给 NVIDIA 看的第一份材料

---

# Stage 7 — 人工验证抽样包（$20）★ 论文可辩护性

数据集当前最大弱点是"无人工真值"（同域前作 308/849 bugs 是 100% 人工标注、κ 0.84–0.9）。
执行机不能代替人判定，但要把人力成本压到最低。

1. 抽样（确定性 seed，方法写入 `audit/sample_manifest.json`）：
   - **A 组 · 卡片 357 张**（5,016 的 95% 置信 / 5% 误差），按 repo × kind ×
     static_detectability 分层
   - **B 组 · 回归对 150 对**（A 级 100 / B 级 50），按 repo 分层
   - 合计覆盖全部 11 个大类（每类 ≥ 5 项）
2. `audit/human_audit_packet.md`（纯 Markdown，离线直读），每项含：机器结论
   （kind/symptom/一句话 mechanism/叶）+ evidence 引文 + 关键 diff hunk（≤40 行）+
   B 组另附两个 commit 的 subject 与归因原文 + 待填空位
   `[ ] 同意   [ ] 不同意 → 正确值: ____   [ ] 无法判断`
3. `audit/human_audit_template.jsonl`（预填 target_id/field/machine_value，
   `human_value`/`verdict` 留空），回填后可被 `apply_audits.py` 直接消费
4. `reports/human_audit_packet.md`：抽样方法、分层表、agreement 与 95% CI 的计算公式

**只准备材料，不产生标签。** 建议 Rui 优先判 B 组。

---

# Stage 8 — 论文素材包（$30）

把散落各处的数字汇成一处，便于直接写论文：

1. `paper/tables/`：所有主表导出为 CSV + Markdown（baseline 对比、消融、per-kind、
   per-taxonomy、分层 FPR、跨仓泛化、数据集统计）
2. `paper/figures_data/`：作图用的数据（漏斗各层计数、taxonomy 分布、
   detectability 分布、对抗验证前后 recall-FPR 曲线、recall@budget 曲线）
3. `paper/claims.md`：**每条可写进论文的主张 + 支撑它的具体文件与数字 + 已知反驳**。
   必须包含以下已知要点：
   - novelty 应表述为"**首个 AI-infra 性能回归（引入→修复）配对数据集**"，
     **不可**表述为"首个 AI-infra 性能数据集"（后者已被 arXiv 2506.09713 的 929-bug
     数据集、2512.20345 的 849 issues、2506.10426 的 308 bugs 证伪，且都公开可下载）
   - 检测器侧须引用 arXiv 2604.00222（2026-04，commit 级性能回归风险预测，
     ROC-AUC 0.694）——它是相关工作，也是一个弱 baseline 参照
   - Tier-A 归因（开发者原话）的证据强度可主张高于算法 SZZ：ICSE 2021 的
     developer-informed oracle 正是同一口径，而最优 SZZ 变体对其仅 61% F-measure
4. `paper/limitations.md`：机器标注/机器仲裁、judge 为 LLM、无 GPU 复现、
   FPR 基于抽样、静态天花板——每条附缓解措施与未来工作

---

## 完成判据（全打勾才算完）

- [ ] S1：泄漏测试 3 项断言通过；指标齐全；judge κ ≥ 0.7 且跨家族校验入报告；子集冻结；memorization 补探完成
- [ ] S2：四条 baseline 全跑完（含 (c) 关键词与 (d) 跨家族）；`baseline_results.md` 出数；tag `baseline-v1`
- [ ] S3：三条腿各有报告与成功判据结论；对抗验证层就位；dev 报告含对抗前后对比；配置冻结；tag `detector-v1-frozen`
- [ ] S4：test 单次跑完 + 四项消融；`detector_v1_results.md` 含主表/消融/诚实小节；tag `detector-v1`
- [ ] S5：跨仓泛化报告出数
- [ ] S6：前瞻试运行报告（含 top 20 人工核验清单）或明确记录"fetch 失败已跳过"
- [ ] S7：抽样包（357 卡 + 150 对）+ 可读评审文件 + 可回填模板
- [ ] S8：paper/ 下三类素材齐全，`claims.md` 含上述三条已知要点
- [ ] 全程 leak_attempt 累计为 0
- [ ] 总花费 ≤ $2,000
- [ ] 更新 `report.md` 总览；最终 git tag `run2-complete`

## 异常处理

- 触分帽 → 停该 Stage，记录已完成比例与单价，继续下一个**不依赖它**的 Stage，最后汇总
- judge κ < 0.7 且两轮修 prompt 无改善 → **停**，等 Rui（评测有效性问题，不能带病前进）
- 某条腿完全失败 → 记为负结果，用其余两条继续，**不放宽门槛**
- Stage 4 前发现配置需要改 → 说明理由后改，但必须**重跑 Stage 3 的 dev 评估**，
  且在报告中记录改动（禁止用 test 反馈驱动改动）
- 发现呈现规则被违反（leak_attempt > 0 或视图含后验字段）→ **立即停**，最高级别问题

## 跑完之后

停下等 Rui。届时项目状态应为：数据集 v0.2 + 检测器 v1（含冻结配置与可复现评测）
+ 完整实验矩阵 + 论文素材 + 待人工判定的抽样包。
下一步（v2 方向、投稿、与 NVIDIA 接触）由 Rui 与规划方决定。
