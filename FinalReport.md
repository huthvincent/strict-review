# bug_detect — 项目总报告

> 截至 2026-07-19（本次目录整理完成时）。每次大更新后由当值 agent 重写本节，
> 旧内容移入文末"历史快照"。

## 项目一句话

给 Megatron-LM 等 AI infra 仓库造比 NVIDIA 现役 `/claude strict-review` 更好的性能
守门员：静态抓得住的当场指出（附历史判例），抓不住的精准路由到已有 perf CI 实测。

## 已完成

**数据集线（v0.2 冻结，花费 ~$4,470）**：四仓 33,089 commit 三层漏斗 → 5,016 张
性能案例卡＋622 对回归配对（A 级 502）＋29,703 负样本＋74 叶/11 大类分类学＋冻结
评测协议。质量：校准闸门 recall 1.000、双标 κ 0.81/0.971、记忆污染 0/934、
完整性 10/10。已知伪影：负样本文件含上万重复 case_id（v0.3 待清理，评测子集已修正）。

**检测器线（RUN2，8 Stage 全完成，花费 ~$3,167）**：
- 对手实测：strict-review overall@2 13.1% / 北极星 regfix@2 14.9% / FPR 17%；
  裸 Opus 12.2% / 14.9% / **5%**——手写 rubric 零召回增益、误报×3（头条发现）；
  keyword 0%（任务不在文字里）；Nova 6.5%/8.0%/23%（排除自我偏好）。
- 冻结版 detector_v1：overall **18.6%**（胜一切）/ regfix 12.3%（负于 baseline）/
  FPR 14.9% / $0.233（比对手便宜 33%）。
- **消融真相（论文核心）**：leg3 独撑召回（去之 −15.1pp）；v1 的 leg1/leg2/对抗层
  在 test 净负；预登记消融 ablate_adversarial regfix **18.0%** 反超一切
  → v2 方向＝leg3 主干＋其余组件重设计。
- 跨仓迁移损失仅 4.8pp；前瞻 61 个未见新 commit 触发 26 个（43%），top-20 待人判。
- judge κ=0.926 / Nova 一致 89%；leak=0（程序化核验）；test 单次开封纪律无违反痕迹。

## 项目方向（2026-07-19 Rui 拍板）

**产品优先，暂缓论文。** 目标就是做出真正可用、真正更好的 strict-review 替代品。
paper/ 素材全部保留，投稿相关事项降级为背景任务。上 HF / GitHub 均为便利
（宪章 5.3/5.4 已相应修订）。

## 验收遗留

1. ~~花费 $3,167 超帽~~ **已闭环**：Rui 2026-07-19 书面确认批准属实，留痕见
   `archive/run2_shipment/DECISION_budget_approval.md`，改判 pass-with-deviation。
2. `paper/` T1/T4 与 `evaluation/reports/baseline_results.md` 判分漂移 2–3 个命中
   （论文暂缓后不紧急；将来使用前统一重判即可，宪章 7.5）。
3. judge 缓存与 git tags 未随包回传，召回数字暂无法离线复现（FPR 已全部独立复算吻合）
   ——下次穿梭捎回。
4. 缺 leg2_tuning / leg3_routing 两份专项报告；leg2 的"胜 baseline(b)≥30%"判据未宣判。

## 待办（产品优先排序）

1. **RUN3 已就绪待执行**：作业书 `RUN3_INSTRUCTIONS.md`（红队修订 25 处后定稿，
   总帽 $450 估 ~$300），已同步 `github.com/huthvincent/GAA/ai_infra_run3.md` 供执行机
   拉取。内容＝detector v2 正式实现与 dev 全量评测（成功判据钉死在旧口径 ≥25.3%）、
   74 叶验证表＋微基准模板、考题框架 McNemar 检验、与 v1 的 61 commit 前瞻对决。
   v2 配方已有本地实证（`detector/v2_prototype/`：引入侧 2/6 vs 裸模型 0/6，$0）。
2. Rui：判 `audit/` 抽样包（质量校准）与 `detector/reports/prospective_run.md`
   top-20（= 给 NVIDIA 的可用性证据，也是 v2 的误报样本来源）。
3. 数据集 v0.3：清理负样本重复伪影 → 更新 `dataset/` 镜像 → 同步 HF。
4. 下次穿梭作业书顺带：judge cache、tag 清单、T1/T4 重判、run2 目录入 git。
5. （暂缓）投稿路线背景见 `paper/`。

## 全项目账本

数据集线 ~$4,470 ＋ RUN2 ~$3,167 ≈ **$7,637**。

---

## 历史快照

（暂无——本文件为整理时初立。）
