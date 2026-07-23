# MegaPerfBench RUN3 — 交付包（检测器 v2 ＋ 叶子验证表 ＋ 考题框架检验）

> 生成 2026-07-23 · git tag `run3-complete`（起始 `run3-start`）· 依据 `ai_infra_run3.md`
> 模型: 一切 LLM = Opus 4.8 标准通道 · judge = eval_judge.v1（跨轮锚检 κ=0.902）
> **总花费实测 ~$222（硬帽 $450）· 全程 leak_attempt = 0**

## 一分钟结论

**detector_v2 是一个诚实的负向结果。** 它未达任何 recall 判据（dev overall@2 **1.3%** ≪ 25.3%，
pair 0%；前瞻 **0/61** vs v1 24/61），但 benign FPR 完美（0%）。根因是**严重过度抑制**——
"手册常驻 + 宁缺毋滥 + 三问 gating" 把强模型推向过度保守，250 项 dev 仅 13 项产出 finding。
**消融证明常驻手册本身是主要拖累**（去手册 recall 2.0%→8.0%）。同时，**考题框架伪影假设被
McNemar 配对检验证伪**（方向与假设相反：修复侧反而更易命中）。详见 `reports/run3_report.md`。

## 目录

```
README.md                     本文件
ai_infra_run3.md              本轮作业书（charter）
report.md                     项目总览（含 RUN3 banner）

reports/
  run3_report.md              ★ RUN3 总报告（各 Stage 数字 + 诚实结论 + Rui 终审事项）
  judge_anchor.md             S0 judge 时漂锚检（κ=0.902 通过）
  tables_rebuild_note.md      S0 T1/T4 cache-only 重建（0 miss）
  framing_report.md           ★ S1 考题框架检验（McNemar×5 方向反假设，证据整理）
  framing_summary.json / framing_diffsize.json / inducing_conversion.json
  handbook_crossread.md       S2 手册语义对读（0/10 重写）
  handbook_leak_audit.md      S2 三资产泄漏审计（0 外源）
  leaf_verification_smoke.md  S2 微基准冒烟（CPU template-only，train 卡）
  gate_killrate.md            S3 画像门 train 误杀率（0.09% < 2%）
  cost_projection.md          S3 成本投影（用滚动结余覆盖，不启用降级阶梯）
  detector_v2_dev.md          ★ S4 dev 全量主表 + 修正口径附表 + 成功判据宣判 + 诚实结论
  detector_v2_ablation.md     ★ S4 κ复检(1.000) + 跨家族(100%) + 2消融带CI（手册消融关键）
  prospective_v2.md           ★ S5 前瞻 v2 vs v1（0/61 vs 24/61）
  detector_v2_summary.json / detector_v2_ablation_summary.json
  git_tags.txt                所有 tag

knowledge/                    S2 三件套（train-only，泄漏审计 0 外源）
  handbook.v1.md / .jsonl     74 叶手册（73 页 + 1 跳过无卡）
  repo_profile.v1.json        4 仓画像（热点文件/模块/排除/Megatron recipe 提示）
  leaf_verification.v1.json   74 叶验证表（15 route-recipe/58 microbench/1 not-verifiable）
  microbench_templates/*.py   58 个 pytest 微基准模板
  gate_killed.json            画像门误杀的 3 个 train 边界案例

detector/detector_v2.config.json   ★ 冻结配置（FROZEN 注记含全部超参与依据）

paper/                        重建的 T1/T4（cache-only 0 miss）+ claims.md/limitations.md
predictions/
  .judge_cache.jsonl          ★ 全部 judge 缓存（含本轮，§7.4 要求）
  detector_v2/                dev_tune.jsonl + dev_tune_inducing.jsonl（转换双跑）
  v2_ablate_handbook|tools/   2 消融预测
  detector_v2_prospective/    61 未见 commit 预测
splits/                       inducing 视图 + 冒烟/消融子集
scripts/                      全部 RUN3 代码（15 个）
```

## 两条不可违反纪律的遵守证据
1. **test 本轮零新模型调用**：S0/S1 的一切 test 重聚合在 `JUDGE_CACHE_ONLY=1` 下跑，miss 抛错；
   T1/T4 重建 0 miss；framing McNemar 全 cache-only。
2. **呈现规则**：全部预测 `leak_attempt` 累计 = 0；引入侧视图仅白名单键，无后验字段。
   一切资产 train-only，泄漏审计 0 外源。

## 等待 Rui 终审事项（详见 run3_report.md）
1. framing 结论与北极星口径是否修改（证据显示框架伪影假设方向相反）。
2. 修正口径是否升级为 v0.3 标准（本轮仅作附表，采纳规则未触发）。
3. 北极星分母 261 对同一回归双计 → 建议 v0.3 去重。
4. v2 方向：消融指向"弱化/移除常驻手册 + 放宽 gating"，或回到 v1 的静态规则+检索路线只加画像门。

## 复现
环境同 RUN2：`cd /mnt/efs/tsfm/rui/skill_opt && ~/.local/bin/uv run --no-project python <script>`，
env `MPB_BASE=<工作目录> MPB_REPOS=/home/ec2-user/megaperf_repos AWS_REGION=us-east-1`。
cache-only 重聚合加 `JUDGE_CACHE_ONLY=1`。数据集输入（cases/pairs/negatives/taxonomy）与四仓 clone
体量大，未含本包，在正本工作目录。
