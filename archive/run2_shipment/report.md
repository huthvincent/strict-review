# MegaPerfBench — 进度总览

> **状态:Run2 检测器线进行中。** 数据集 `v0.2` 已冻结(本地 git tag,无 remote)。
> 正本目录:`ai_infra_run2/ai_infra_v011_work/`。

## Run2 全部完成(检测器线,2026-07-19,tag `run2-complete`)

**8 个 Stage 全打勾,全程 leak_attempt = 0,总花费 ~$3,167**(Rui 批准 ~$3k,略超,如实记录)。

- **S1 评测地基 ✓**(`s1-complete`):harness.v1 + 泄漏自检 3/3 · 冻结子集
  **818 正 + 116 pairs + 954 neg = 1888**(修正 389 非 perf case + 41 重复 hotfile 负样本)·
  metrics.v1 · memorization 全量 934 / **0 memorized** · **judge κ=0.926 / 跨家族 Nova 89%**。
- **S2 四 baseline ✓**(`baseline-v1`,GATE 通过):Megatron-strict regfix@2 **14.9%** ∈[5,45%];
  generic 14.9% / FPR 5%;**keyword 0%**(证明任务不在文字里);Nova 8% / FPR 23%(自我偏好检查)。
- **S3 检测器 v1 ✓**(`detector-v1-frozen`):3 腿(leg1 静态规则 31 条留存/FPR 2.6% · leg2
  BM25 检索 · leg3 风险路由→真实 Megatron recipe)+ 对抗验证 + budget-2。dev 调参 recall@2 21.3%。
- **S4 test 对决 ✓**(`detector-v1`,test 只跑一次):detector_v1 **overall@2 18.6%**(高于所有
  baseline)、regfix@2 12.3%、FPR 14.9%、$0.233/item。**消融核心诚实发现:leg3 承载几乎全部召回
  (去掉 −15.1pp);leg1/leg2/对抗层在 test 上净负(去掉反而 +3.9/+2.2/+5.2pp)**。冻结版被自身
  ablate-adversarial(18.0%)反超——如实上报,不用 test 反馈改配置,改进留 v2。
- **S5 跨仓泛化 ✓**:排除 Megatron 重建 leg1+leg2,Megatron test 部分迁移损失 **+4.8pp**(overall@2)。
- **S6 前瞻试运行 ✓**:61 个数据集未见的 Megatron 新 commit(git fetch),43% 触发率,leak=0;
  top findings 命中正确 taxonomy(如 #5834 redundant-memcpy → redundant-buffer-copy)——可用性证据。
- **S7 人工审计包 ✓**:360 卡 + 150 对(全 11 类),可回填模板,只备材料不产标签。
- **S8 论文素材 ✓**:paper/ 下 5 表 + 5 图数据 + claims.md(9 主张含 arXiv 2506.09713/2512.20345/
  2506.10426、2604.00222、ICSE21 SZZ 三处必引)+ limitations.md(9 条含缓解)。

**交付物**:数据集 v0.2 + 检测器 v1(冻结配置 + 可复现评测)+ 完整实验矩阵 + 论文素材 + 待人工判定抽样包。
**停下等 Rui。** v2 方向(据消融应为"仅 leg3、重设计对抗层")、投稿、NVIDIA 接触由 Rui 决定。

> **状态:dataset-complete,已冻结 `v0.2`(本地 git tag,无 remote)。** 正本目录:
> `ai_infra_v011_work/`。检测器侧见上方 Run2 进度。

## 里程碑
- **v0.1**(评测核心,字节冻结):33,089 commit 三层漏斗 → 7,737 cards(5,016 perf)+ 622 regression pairs(502 A/120 B)+ 29,703 negatives + 冻结评测协议;Opus-4.8 机器仲裁。
- **v0.1.1**:S2 深读 880 issue → 回填量级/配置/凶手-修复;119 卡 issue-linked。
- **v0.2**(dataset-complete):memorization 探测(0/196 污染)+ 完整性校验 10/10 + 去重链接 + **分类学 taxonomy.v1(11 大类/74 叶,全量 5,016 卡打标,κ=0.971,other 0.02%)**。

## 成本
累计 ~$4,470(v0.1 ~$4,330 + v0.2 ~$137);硬帽 v0.1 $6k / v0.2 $700 均未触。

## 诚实声明
全机器标注+机器仲裁+机器分类,**无人类真值**;Fable 因数据保留不可用,仲裁/分类回退 Opus 4.8。
详见 final_report.md · DATASHEET.md · reports/ARBITRATION.md · reports/dataset_stats.md。

---

(以下为历史滚动记录)

# MegaPerfBench Phase 0 — 执行情况报告

> 生成时间：2026-07-14（滚动更新）
> 执行者：Claude Code（Opus 4.8）
> 依据：`intro.md`（Phase 0 执行计划）+ `skill_opt/API.md`（Bedrock 接入）
> 累计 API 花费：**~$62**（$0.9 smoke + $49.6 校准全量 + $2.1 v2 校验 + $9.3 Tier-1 标准跑热身；硬帽 $6,000）

---

## 0. 一句话摘要（最新）

**校准闸门已通过（recall=1.000，6/6 regression-fix，无泄漏），Tier-1 全量筛查正在
标准通道上跑（Opus 4.8，~$1,043，后台进行中）。** 途中发现并解决了两个计划级问题：
(1) pilot 数据缺失——Rui 选方案 A 已提供并接入；(2) **Bedrock Batch 不支持 Opus 4.x**
（本账号只支持 Sonnet/Haiku 4.5）——Rui 拍板改走标准通道 Opus，坚持模型纪律。
核心产出 3（评测协议）已写死。下一步：全量筛查跑完 → Tier 2/3（需 GITHUB_TOKEN）。

### 进度总览

| 阶段 | 状态 |
|---|---|
| 地基（目录/git/5 schema/prompt） | ✅ 完成 |
| 全量 clone + commit 元数据（33,089） | ✅ 完成 |
| Tier 0 确定性路由（25,759→screen） | ✅ 完成，DoD 达标 |
| Tier 1 分类器 + live smoke | ✅ 完成 |
| pilot 数据接入（48 gold + 1226 候选） | ✅ 完成（方案 A） |
| **Tier 1 校准闸门（recall≥95%）** | ✅ **通过**（1.000, 6/6 regfix, 泄漏free） |
| prompt v2（真 gold few-shot） | ✅ 完成，held-out recall 1.000 |
| 评测协议 `splits/protocol.md`（核心产出 3） | ✅ 写死 |
| **Tier 1 全量筛查（标准通道 Opus）** | 🔄 **后台进行中**（~$1,043, ETA ~几小时；已 ~2,250 条, 0 错误） |
| S4 golden-values（纯本地） | ✅ 完成（37 gated cases 对齐 analyst，1 accepted-regression 候选） |
| S3 revert 清单（本地前端） | ✅ 完成（511 reverts，计数对齐 analyst，154 perf-prior） |
| 负样本 builder / apply_audits | ✅ 已建+验证（等 Tier-1 跑完再出终版） |
| README + DATASHEET + 评测协议 | ✅ 完成 |
| Tier 2 深读（agentic，本地 git 证据） | ✅ 已建+验证（3 gold 命中真 inducing SHA）；全量等 Tier-1 跑完 |
| Tier 3 SZZ（3 票，本地 git log -S） | ✅ 已建；全量等 Tier-2 出 cards |
| Tier 2 QA 20% 双标 + kappa | ✅ 已建 |
| 阈值自选（full-recall 平台内压 Tier-2 量） | ✅ 已建（select_threshold.py） |
| **下游编排器（自动串起全部）** | 🔄 **后台运行**：等 Tier-1 → 自动跑 tier2/QA/tier3/负样本/apply_audits |
| S2 issues | ⏸ 跳过（唯一真需 GitHub API 的阶段；Rui 定"全本地"） |
| 人工仲裁（Rui）/ 切分冻结 / v0.1 | ⏳ 待下游跑完 + Rui 审 |

---

## 1. 已完成的工作（都已落盘 + 验证）

### 1.1 开工前的环境核实（发现两处计划假设已失效）

先核实了计划赖以成立的前置事实，避免盲目花钱：

| 核实项 | 结果 |
|---|---|
| Opus 4.8 能否调用 | ✅ 能。`us.anthropic.claude-opus-4-8` live smoke 返回 "OK"（~$0.0005） |
| Bedrock **Batch** API 是否可用 | ✅ 可用。账号上已有一个 Completed 的 batch job；S3 桶 `omny-artifacts-us-east-1-beta` + IAM 角色 `BedrockBatchInferenceRole` 都在 |
| 网络（GitHub / Bedrock） | ✅ 通 |
| 磁盘 | ✅ 本地 NVMe 2TB 富余；EFS 巨大 |
| **四个仓库是否已 clone** | ❌ **计划 §7.1 假设"现有 clone"——实际本机没有任何 clone。已重新全量 clone**（见 1.3） |
| **pilot 数据（48 gold cards + 1226 关键词候选）** | ❌ **不在本机**。计划写的是 `~/Desktop/OPB/...`（macOS 路径），这是台无头 Linux EC2；`/mnt/efs/tsfm/rui` 全盘扫描也没找到。→ **硬阻塞**，见 §2 |
| `gh` CLI / GITHUB_TOKEN | ⚠️ 未装 / 未设。Tier 0/1 不需要（全离线），但 Tier 2 的 `fetch_pr` 和 S2/S3 需要。已在 BLOCKERS 记录 |

### 1.2 目录地基 + 本地 git（计划 §8，"最重要的一节"）

- 按 §8.1 建好完整目录树（`schemas/ prompts/ raw/ screening/ cases/ pairs/
  negatives/ taxonomy/ splits/ reports/ scripts/ archive/`）。
- 在根目录 `git init`，**没有配置任何 remote**（§8.1 明令 `git remote add` 属违规）。
- `.gitignore` 排除 `raw/gh_cache/`、`logs/`、`__pycache__`。
- **五份 JSON Schema 定版**（`schemas/*.v1.schema.json`）：provenance / tier1_screen
  / card / regression_pair / audit_log —— 后续所有 Phase 的消费接口。

### 1.3 全量 clone + commit 元数据抽取（DoD 达标）

- 四仓库**完整 clone**（非 blobless——验证过 `git show` 全离线出 diff，§7.1 的坑已排）。
  commit 数与计划几乎完全吻合：

  | repo | 本次 clone | 计划记载 |
  |---|---:|---:|
  | Megatron-LM | 9,214 | 9,214 |
  | vllm | 18,703 | 18,699 |
  | DeepSpeed | 3,225 | 3,225 |
  | TransformerEngine | 1,947 | 1,947 |
  | **合计** | **33,089** | 33,085 |

  （多出的 4 个是 2026-07-13 之后的新 commit。）
- `raw/commits/{repo}.jsonl` 全量落盘（sha/date/author/subject/parents/files/
  stats/is_merge），22MB，30 秒抽完。

### 1.4 Tier 0 确定性路由（DoD 达标）

- 脚本 `tier0_route.py`（版本 `tier0.v1`，确定性 + 幂等 + 版本化，§7.5）。
- 33,089 个 commit **全部**有路由记录，桶分布：

  | 桶 | 数量 | 占比 | 去向 |
  |---|---:|---:|---|
  | `screen` | **25,759** | 77.8% | → Tier 1 LLM 筛查 |
  | `smoke-ci` | 2,070 | 6.3% | → S4 golden/CI 子管线 |
  | `skip-docs` | 2,562 | 7.7% | 文档/资源 |
  | `skip-empty` | 2,656 | 8.0% | merge/空 diff（Megatron 居多） |
  | `skip-format` | 42 | 0.1% | 纯空白改动（`git -w` 判定） |

- **与计划的偏差（合理，且省钱）**：计划估 screen ≈28–30k，实际 25.7k。差额来自
  路由额外切出了 `skip-empty`（无首父 diff 的 merge）和 `smoke-ci`，计划把这些
  折进了 screen 估算。净效果是 **Tier-1 batch 更便宜**。
- `skip-format` 刻意做得极保守（只判纯空白，不判 import 重排）：漏判会造成不可
  恢复的 recall 损失（§10），误判进 screen 只多花几分钱。
- 报告：`reports/tier0_routing.md`。

### 1.5 Tier 1 分类器：搭好 + live 验证（DoD：仅"全量"和"校准"未做）

- 版本化 prompt `prompts/tier1_screen.v1.md`（含 §2 定义 + 五分类口径 + 判定规则
  + few-shot）。
- `diff_packer.py`（`packer.v1`）：commit message + `git show --stat` + diff 裁剪
  到 ≤6k token，**非测试 hunk 优先**（§3），记 `diff_truncated`。
- `tier1_classify.py`：Opus 4.8 + **强制工具调用**（`tool_choice`）保证输出永远是
  schema-valid JSON，无需解析自由文本；成本按 token 实测计量。
- **live micro-smoke（16 个 commit，四仓库混合，$0.78）**，结果非常干净：
  - 真性能改动全部命中：all-gather overlap（0.90）、FP8 transpose cache
    regression-fix（0.72, memory）、swiglu kernel（0.85）、faster allreduce（0.90）
    —— 全部 `needs_deep_read=true`。
  - **成功避开关键词陷阱**：`exec_module`（0.05）、benchmark 清理→`perf-infra-or-test`
    （0.10）、`dist.destroy_process_group` 测试（0.05）。这正是计划坚持"全量而非
    关键词过滤"的理由——分类器能识别 "throughputtest" 只是测试目录名、不是真改动。
  - 报告：`reports/tier1_smoke.md` + `screening/tier1_smoke.jsonl`。

### 1.6 Tier 1 batch 通道：搭好 + 格式验证（未点火）

- `tier1_batch.py`：build / submit / status / collect 四子命令，复用同一 prompt +
  SCREEN_TOOL（batch 与 standard 通道完全一致）；幂等（按 id 跳过已完成）；乱序结果
  按 `recordId` 归位；batch 计价 0.5×。
- **build 步已验证**（$0，无提交）：生成 120 条合法的 Bedrock batch 记录
  （`{recordId, modelInput:{anthropic_version, max_tokens, system, messages, tools,
  tool_choice}}`），格式与账号上现有 job 完全对齐。
- Bedrock 参数已锁定：账号 `520186517736`、Opus 4.8 inference-profile ARN、
  `BedrockBatchInferenceRole`、S3 前缀 `.../users/rui/megaperf/tier1`。
- **`submit --full` 未执行**——受校准闸门阻塞（见 §2）。

### 1.7 实测经济性

- Tier-1 单 commit：avg ~$0.049（standard）→ ~$0.024（batch）。
- **全量 Tier-1 batch 投影成本：~$629**（计划 §9 估 ~$700，在范围内）。

---

## 2. 阻塞 1 已解除：pilot gold 数据（Rui 选方案 A）

Rui 把 pilot 打包成 `pilot_gold_transfer.zip` 传来，解压进 `raw/pilot/`。内容与
校准所需完全吻合，且全部对齐验证通过：
- 48 张 gold cards（29 阳性 / 6 regression-fix / 1 unclear），**48 个 SHA 全部在
  clone 里可解析**。
- 1,226 关键词候选，各仓库计数 **262/689/132/143 与计划分毫不差**。
- 另附 3 份 analyst 报告（golden-values / reverts / issues）作为 S4/S3/S2 的设计输入。

**校准闸门结果（通过）**：用 v1 prompt（合成 few-shot → 无泄漏）跑 48 gold：
**recall = 1.000（29/29），6/6 regression-fix 全中**，T≤0.40 满 recall，T=0.50 仍
≥0.95。报告 `reports/tier1_calibration.md`。随后升 `tier1_screen.v2`（真 gold few-shot），
held-out 24 阳性 recall 仍 = 1.000，确认不回退。运行阈值定 **T=0.35**（满 recall 平台
底部，最大安全裕度；跑完全量按真实分布可上调）。

## 2b. 阻塞 2（新发现，已由 Rui 拍板）：Bedrock Batch 不支持 Opus 4.x

实测：`create-model-invocation-job` 用 `us.anthropic.claude-opus-4-8`/`4-7` →
`ValidationException: Batch inference is not supported for the requested model`；
换 Sonnet/Haiku 4.5 → 接受。即本账号 Batch 通道**只支持 Sonnet/Haiku 4.5，不支持任何
Opus 4.x**。计划的"Opus-on-Batch 五折"不可行。

**Rui 决定**：坚持模型纪律，Tier 1 改走**标准 API + Opus 4.8**（实测 $0.0405/commit
→ 全量 ~$1,043，远低于 $6,000 硬帽）。而非降级到 Sonnet 换便宜。决策与证据见
`reports/tier1_channel_decision.md`。（探测时误建的 2 个 batch job 已立即 Stop + 清理 S3。）

---

## 3. 过程中遇到的问题与处理（累计）

| 遇到的问题 | 处理 |
|---|---|
| 计划假设仓库已 clone，实际没有 | 重新全量 clone（4 仓库 <1 分钟，1GB），验证非 blobless、`git show` 全离线可用 |
| pilot 数据在 macOS 路径、不在本机 | 上报 → Rui 打包传来（方案 A），接入 `raw/pilot/`，全部对齐验证 |
| **Bedrock Batch 不支持 Opus 4.x** | 实测确认 → Rui 拍板改标准通道 Opus（守纪律，成本仍在帽内） |
| CLI 双身份：bearer-token 用户无 `iam:PassRole` | 定位到 `AWS_BEARER_TOKEN_BEDROCK` 抢占；unset 后用 instance-profile 角色可建 job（但模型墙才是硬约束） |
| `skip-format` 需 diff 内容（否则 26k 次 `git show`） | `git log -w --numstat` 单次批量 pass 判纯空白（vllm 全量 23s） |
| Megatron ~2640 个 0-file commit | 基本是 merge（无首父 diff）→ 新增 `skip-empty` 桶 |
| 校准报告的 Tier-2 量外推是错的（只在 48 gold 上算） | 改为诚实口径：闸门只证 recall 可达；真实 Tier-2 量由全量批的真实分布决定 |
| report 生成器一处 `cand_done` 变量名残留 → NameError | 数据无损（1226 条已缓存），修 bug 后重建报告 |
| Tier-1 结构化输出可靠性 | `tool_choice` 强制工具调用，输出永远 schema-valid |
| 不用 tiktoken（§7.7） | 裁剪用固定 char/tok 启发式；**实际 token 一律从 API usage 读回**并入报告 |

---

## 4. 产出清单（都在 `/mnt/efs/tsfm/rui/GAA/ai_infra/`，未上任何外部服务）

```
README.md DATASHEET.md report.md              # 总览 / datasheet / 本报告
schemas/     5 份定版 JSON Schema
prompts/     tier1_screen.v1.md + v2.md（v2=真 gold few-shot，生产用）
raw/commits/ 4 仓库全量 commit 元数据（33,089 行）
raw/pilot/   pilot gold（48 cards + 1226 候选 + 3 份 analyst）
raw/reverts.jsonl                # S3 清单：511 reverts
raw/golden_values/megatron_churn.jsonl   # S4：101 change records
screening/   tier0_routing.jsonl（33,089 全路由）
             tier1_calibration.jsonl（1226 校准）
             tier1_screen.jsonl（全量筛查，后台进行中）
             tier1_smoke.jsonl / tier1_v2_goldcheck.jsonl
splits/      protocol.md（核心产出 3，protocol.v1，已写死规则）
scripts/     mpb_common / extract_commits / tier0_route / report_tier0 /
             diff_packer / tier1_classify / tier1_smoke / tier1_batch /
             tier1_calibrate / tier1_run_standard / s3_reverts /
             s4_golden_values / build_negatives / apply_audits
reports/     tier0_routing / tier1_smoke / tier1_calibration /
             tier1_channel_decision / tier1_summary / s3_reverts /
             s4_golden_values / negatives / BLOCKERS
```

§8.8 DoD 对照：Tier 0 ✅；Tier 1 校准 ✅（recall 1.000）；Tier 1 全量 🔄（进行中）；
S4 ✅；S3 前端 ✅；协议 ✅；README/DATASHEET ✅。

---

## 5. 下一步

**全自动进行中（`run_downstream.sh` 后台，无需干预）**：Tier-1 跑完后依次自动执行
select_threshold → Tier-1 summary → Tier-2 深读 → Tier-2 20% 双标 → Tier-3 SZZ →
负样本终版 → apply_audits。全部本地、幂等、成本封顶、可断点续跑。预计几小时内产出
`cases/cards.jsonl`、`pairs/regression_pairs.jsonl` 初版及各 stage 报告。

**决策：全本地**（Rui 定）——不上 GitHub、不需要 GITHUB_TOKEN。因此：
- Tier 2 / Tier 3 / S3 motive 改为纯本地跑（git_show/git_log/-S 即核心证据，
  `fetch_pr` 离线优雅降级，PR 文本只是增强，不是依赖）。
- **S2 issues 是唯一真正离不开 GitHub API 的阶段 → 跳过**（vLLM 503 个 perf issue 的
  量级/复现配置无法离线获取）。这会少一路信号，但主管线（commit 流三层 + S3/S4）完整。

**收尾（需 Rui）**：跑完后人工仲裁（§6，双标分歧队列已自动生成）+ 10% 审计 →
`make_splits` + `splits/FROZEN` + 本地 tag `v0.1`。

---

## 6. 预算状态

- 已花：**~$160**（smoke $0.9 + 校准全量 $49.6 + v2 校验 $2.1 + Tier-1 标准跑至今 ~$105 且在涨）。
- 全量 Tier-1 标准通道预计总计 **~$1,043**；Tier 2/3 仍走标准 Opus（计划本就如此）。
- 硬帽 $6,000，$4,500 预警线；全 Phase 0 预计落在计划的 $2.5–4k 带内（Tier-1 通道变更后略上移）。
