# MegaPerfBench RUN4 — detector v2.1（按契约实现并重考）交付包

> 生成 2026-07-23 · git tag `run4-complete`（起始 `run4-start`）· 依据 `ai_infra_run4.md`（run4.v3）
> 模型 Opus 4.8 · judge eval_judge.v1（锚检 κ=0.870/复检 0.898/Nova 94%）
> **实测 ~$138（硬帽 $500）· 全程 leak=0 · route 不变量=0 · dev error 0**

## 一分钟结论

**v2.1 召回大翻身、FPR 未达标。** 按 CONTRACT A–E 逐字实现（修复 RUN3 v2.0 四处缺陷）后：
dev overall recall@2 **77.3%**（v1 21.3%，判据 25.3% → 达标）、pair@2 **26.7%**（判据 13.3% → 达标），
但 weighted FPR **23%** > 研究门 15%（产品线 10% 亦未达标）。**2/3 判据达标，FPR 未达标，如实上报。**
消融证明 **route 独立腿正是修复点**（去掉它召回 80%→22%，回到 v2.0 死形态）。

## 本轮关键转折：BLOCKERS → 裁决 → 重考

按契约实现后，**原 v2 口径下活性门只 3/18**——独立审计确认为**契约内在矛盾**
（unverified×0.7 折扣 × route conf≤0.8 × conf_final≥0.5 阈值相乘不可达）。执行机按纪律 #4
**停机 BLOCKERS**（`reports/BLOCKERS_run4.md`）。Rui 裁决：**计分回归 metrics.v1**
（召回=budget2 severe 无 conf 门槛；FP=severe∧conf_raw≥0.5；conf_final 仅排序），charter→run4.v3。
重考后活性门 18/18 过 → FROZEN → 完成 S2–S5。

## 目录

```
README.md / ai_infra_run4.md / report.md
reports/
  run4_report.md              ★ 总报告（各 Stage 数字+成本对账+四方对照+判据宣判+Rui 事项）
  BLOCKERS_run4.md            BLOCKERS 记录（契约矛盾诊断）
  contract_selfcheck.md       CONTRACT A–E 每条→代码行自查（独立审计=faithful）
  liveness_gate.md/json       活性门（run4.v3，18/18 过）
  judge_anchor_run4.md        时漂锚检（96.7%/κ=0.870，含分歧方向）
  detector_v2_1_dev.md/json   ★ S2 主表（77.3%/26.7%/23%）+ 哨兵行 +15.7pp + route/deep 两列
  v2_1_kappa_xfam.md/json     κ 复检 0.898 + Nova 94%
  v2_1_ablation.md/json       ★ 消融：full 80% vs −route 22%（route 腿是修复点）
  replay_v2_1.md/json         S4 诊断回放（28/61，非前瞻，含 bcf4c8fb 标注）
  prospective_new_v2_1.md/json ★ 新前瞻窗（22 未见 commit，v2.1 触发 11/22，真前瞻）
  overlap_sensitivity.md      §2.5 全量 vs 剔除 smoke 重叠（76.6% 仍达标）
  usage_ledger.jsonl          ★ 逐调用 token/cost（检测器 $127.49 精确）
  git_tags.txt
detector/detector_v2_1.config.json   ★ 冻结配置（CONTRACT + run4.v3 口径注记）
predictions/
  .judge_cache.jsonl          全部 judge 缓存
  detector_v2_1/dev_tune.jsonl  dev 250 预测（含 conf_raw + conf_final + unverified + meta）
  v2_1_ablate_route/          消融预测（--ablate route）
  detector_v2_1_replay/       61 回放
  detector_v2_1_newwindow/    22 新前瞻窗
scripts/                      detector_v2_1.py（CONTRACT A–E）+ runner（--ablate route 开关）+
                              usage_ledger + 全部评分/报告脚本 + 共享 harness/metrics
splits/ prospective/          冒烟/消融子集 + 新前瞻窗 commit 清单
```

## 契约遵守证据（§完成判据）
- CONTRACT A–E 逐条→代码行：`reports/contract_selfcheck.md`（独立子代理审计=faithful，仅四处机械替换）。
- 活性门四条全过（run4.v3）：18/18 severe 开火、1/7 neg FP、route 不变量 0、leak 0。
- 全程 leak=0、route 不变量=0、dev error 0；总花费 ~$138 ≤ $500。

## 成本对账（诚实）
- **检测器 side ledger 精确 = $127.49**（`usage_ledger.jsonl`，1246 次调用逐条记录）。
- **判分 side ~$10 为估算**：judge 走 `eval_metrics._opus_judge`（路径早于 ledger 未接入）。
  这是本轮已知对账缺口——但来源明确（judge 路径未接 ledger）、量级已界定，比 RUN3 的 $34 无解释差额透明。
  建议 v2.2 把 judge 也接入 ledger。

## 等 Rui 终审（详见 run4_report.md）
1. FPR 23%>15% 如何处置（接受召回突破 / route 腿加精度过滤 / 其它）。
2. 哨兵行 +15.7pp（负样本更倾向 conf_raw<0.5 severe，压 conf 嫌疑，已标注）。
3. route-HIT 叶子一致率 61%（分诊分类精度 v2.2 改进点）。
4. judge 接入 ledger 消除对账估算。
5. 新前瞻窗 11/22 是本轮唯一合法前瞻信号，可作对外可用性证据起点（需更大窗复核）。

## 诚实声明
- v2.1 FPR 未达标，如实上报；S1 冻结后未用 dev/test 反馈回调。
- 判分为 LLM（κ 复检 0.898 + Nova 94%）；非人工真值。
- §4.1 回放为诊断非前瞻（v2.1 设计参考过该窗，`reports/replay_v2_1.md` 已强制标注）；§4.2 新窗才是真前瞻。
- Mac-40 `views3.json` 未传入（在 Rui Mac），无法本地核对 Mac 口径 / 剔除 Mac 重叠——如实标注为已知缺口。
