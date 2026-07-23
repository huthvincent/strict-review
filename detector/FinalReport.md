# detector/ — 现状总报告

> 截至 2026-07-19（RUN2 验收完成）。

## 状态

**v1 已冻结、已考试、结果已验收。** 正式成绩（`reports/detector_v1_results.md`，
test 单次开封）：overall recall@2 **18.6%**（高于全部 baseline）、北极星
regression-fix@2 **12.3%**（低于两条 Opus baseline 的 14.9%）、FPR 14.9%、
$0.233/item（比 strict-review 便宜 33%）。诚实混合结果。

## 消融揭示的组件真相（v2 的设计依据）

- **leg3（风险路由）独撑召回**：去掉它 18.6% → 3.5%（−15.1pp）。
- leg1（31 条静态规则，dev FPR 2.6% 达标）与 leg2（BM25 检索）在 test **净负**
  （挤占 budget=2 名额；前瞻运行中 leg1 还有匹配测试夹具代码的可见误报模式）。
- 对抗验证层 dev→test 泛化失败（降 FPR 收益 7pp→2.8pp，召回代价 −5.2pp 不变）；
  预登记消融 ablate_adversarial：overall 23.9% / regfix **18.0%**（反超一切，合法可引）。
- **v2 结论：leg3 为主干；leg1/leg2/对抗层重设计后才能回来。**

## 其他实验

- 跨仓泛化（`reports/split_repo_holdout.md`）：排除 Megatron 知识后在 Megatron test
  上 overall 12.4%→7.6%（损失 4.8pp）；regfix 不变——主要因为 leg1/leg2 本来贡献小。
- 前瞻试运行（`reports/prospective_run.md`）：61 个数据集未见的 Megatron 新 commit，
  26 个被标记（43%，作为门禁偏高，v2 要压）；top-20 清单待 Rui 核验
  ——其中机制级 finding（如 #5607 默认路由策略三处齐改、#5349 每 microbatch 阻塞
  all-gather）是给 NVIDIA 的 demo 素材。

## RUN4：v2.1 成为现役检测器（2026-07-22 验收）

冻结口径 recall 77.3%/pair 26.7%/FPR 23%（2/3 判据达标，FPR 原判维持）；
**工作点裁决：展示门 conf_raw≥0.6 → 43.3% / 13.3% / 2%——三轴全面超 v1，
产品红线 10% 大幅达标**。消融坐实 route 独立腿（去之 80%→22%）；哨兵复判=校准
非博弈（分离度 0.866）；新前瞻窗触发 11/22（合法可用性证据起点）。
详见 `RUN4_ACCEPTANCE_AND_OPERATING_POINT.md`；v2.2 小改清单同文件第五节。

## RUN3：v2.0 执行机全量（2026-07-23）——判负，尸检完毕

冻结版 detector_v2.0 在 dev 全量上 overall@2 1.3% / pair 0% / FPR 0%（判据 ≥25.3%
全未达标），前瞻 0/61。**尸检（`v2_autopsy.md`）：四处实现缺陷叠加致哑火（leg3
降级成装饰、三问纪律成 severity 硬闸捂嘴 15/19 条 finding、手册白名单效应、
大 commit 子审被砍到 2 轮），不是配方证伪。** 判分无罪（锚检 κ=0.902）。
v2.1 处方五条已开（复活 leg3 为独立腿、三问改校准器、手册工具化、大 commit 两级审、
冻结加活性门）。RUN3 副产品：`run3_results/knowledge/`（74 叶手册＋验证表＋58 模板）
资产合格可复用。考题框架假设同时被证伪——RUN2 基准与全部历史数字站得住。

## v2 原型烟雾测试（2026-07-19，本地，$0）

`v2_prototype/`：v2 配方（手册常驻＋仓库画像＋父快照上下文核验）在 dev 小样上
与裸模型对照——**引入 commit 场景 2/6 vs 0/6**（含 1 个低可检），误报均 0/4；
另发现 benchmark 考题框架伪影（regfix case 用修复 commit 提问导致方向错位 MISS，
可能压低了 RUN2 全体选手的北极星分）。详见 `v2_prototype/SMOKE_RESULTS.md`。

## 已知欠账

leg2_tuning / leg3_routing 两份专项报告缺失；leg2 的"胜 baseline(b)≥30%"判据未宣判；
judge 缓存未回传（召回数字暂无法离线复现，FPR 已独立复算全部吻合）。
