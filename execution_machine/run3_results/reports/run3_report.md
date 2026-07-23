# RUN3 总报告 — 检测器 v2 ＋ 叶子验证表 ＋ 考题框架检验

- 生成: 2026-07-23 · 对应 git tag `run3-complete`（起始 `run3-start`）
- 依据: `ai_infra_run3.md`（run3.v2 charter）· 工作根: `ai_infra_run2/ai_infra_v011_work/`
- 模型: 一切 LLM = Opus 4.8 标准通道 · judge = eval_judge.v1（跨轮锚检 κ=0.902）
- **总花费实测 ~$222**（硬帽 $450，分帽合计 $410，估算 $300）· **全程 leak_attempt = 0**

---

## 一句话结论

**detector_v2 是一个诚实的负向结果**：它未达任何 recall 判据（dev overall@2 **1.3%** ≪ 25.3%，
pair 0%，前瞻 0/61 vs v1 24/61），但 FPR 完美（0%）。根因是**严重过度抑制**——"手册常驻 +
宁缺毋滥 + 三问 gating" 配方把强模型推向过度保守，250 项 dev 仅 13 项产出 finding。**消融证明
手册本身是主要拖累**（去手册 recall 2.0%→8.0%）。与此同时，**考题框架伪影假设被配对检验证伪**
（方向与假设相反）。两个结果都为 v0.3 / 后续轮次提供了干净的方向。

---

## 各 Stage 关键数字

### Stage 0 — 地基（实测 ~$3）
- cache-only 开关（`JUDGE_CACHE_ONLY=1`）落地，miss 抛错不实时判分。
- git tag `run3-start`；`reports/git_tags.txt` 归档。
- T1/T4 从持久化 judge cache 重建，**0 miss**（`tables_rebuild_note.md`）。
- **judge 时漂锚检通过**：一致率 96.7% / κ=0.902（≥0.8）→ 跨轮对比可比。
- 附带修 bug：`_load_disk_cache()` 并发竞态（伪 cache-miss），加双检锁。

### Stage 1 — 考题框架检验（$0，全 cache-only；证据整理，不下最终结论）
- case↔pair 映射：test 116 对 fix-case **100% 重合** → McNemar 有效。
- **McNemar×5 选手：方向与假设相反**——全部 `c>b`（修复侧命中/引入侧漏 更常见）；
  显著者（detector_v1 p=0.0037、Megatron p=0.0011）方向为**修复侧更易**。
  预登记采纳规则命中 **0/5** → **旧口径为主，修正口径仅附表**。
- 备择解释：case/pair 组 detectability、repo 分布**完全相同**（同一批回归）；但 inducing
  diff **大 8.6×**（真实混杂，方向利于修复侧更易读）。
- 引入侧视图转换率 **135/136 (99.3%)** → `splits/dev_regfix_inducing_views.jsonl`。
- **v0.3 建议**：北极星分母 261 对同一回归双计（fix-case + pair），建议去重为每回归一次。

### Stage 2 — v2 三件套资产（实测 ~$11，全 train-only）
- **手册** 74 叶（73 页 + `weight-transfer-sync` 无 train 卡如实跳过）。
  语义对读 **0/10 需重写**（概念重叠 0.9–1.0；首版词面 metric 误报已修正为 LLM 语义比较）。
- **Repo-Profiler** 4 仓（热点文件/模块/排除表/Megatron recipe 提示）。
- **叶子验证表** 74 叶（15 route-recipe / 58 microbench / 1 not-verifiable）+ 58 微基准模板。
- **泄漏审计：三资产外源引用合计 0**。无 GPU → 2 个 CPU template-only 冒烟（train 卡）。

### Stage 3 — detector_v2 实现（实测 ~$14，冻结）
- 画像门 train 误杀率 **0.09%（3/3386，全 examples/ 边界）** ≪ 2% → 无需收窄。
- 大 commit 分解成本修复（子组 max_turns=2 → big-commit $1.83→$1.00，重跑冒烟）。
- 冒烟 25 项：leak 0 / gate 1 / err 0 / $0.49/item。配置 FROZEN。
- **成本投影 ~$246 vs $200 分帽 → 用滚动结余覆盖，不启用降级阶梯**（书面留痕 `cost_projection.md`）。

### Stage 4 — dev 全量评测（实测 ~$130）★ 主实验
| 指标（旧口径，与 RUN2 同视图同 judge） | v2 | v1 冻结 | v1 最强 | 判据 | 达标 |
|---|---|---|---|---|:--:|
| overall recall@2 | **1.3% (2/150)** | 21.3% | 25.3% | ≥25.3% | ✗ |
| pair@2 | **0% (0/30)** | 13.3% | 10.0% | ≥13.3% | ✗ |
| weighted FPR | **0% (0/100)** | 12% | 19% | ≤15% / ≤10%产品 | ✓/✓ |
- 根因：过度抑制（13/250 产出 finding；4 important / 0 critical / 15 suggestion）。
- **消融归因**：v2−手册 recall@2 **8.0%** vs full **2.0%**（CI 重叠但方向明确）→ **手册主要拖累**；
  v2−父快照工具 无差异。归因边界：画像门/大 commit 分解未单独隔离。
- 判分可信：κ 复检 **κ=1.000 / 100%**（52 项，因 recall 低 pos×HIT 分层仅 4 项，样本 <100 如实记录）；
  跨家族 Nova **100%**。低分非判分问题。
- 修正口径 overall@2 同为 1.3%（碰撞 3，去重敏感性 1.4%）→ 框架效应不能解释低分。
- 画像门直接判 no_issue 的正样本：**0**（门未伤 recall）。

### Stage 5 — 前瞻对决（实测 ~$30）
- 同一 61 未见 commit：**v2 触发 0/61 vs v1 24/61**（15 个被画像门排除）→ 过度沉默在真实新
  commit 上复现。v1 独有 24 触发全部列为诊断。leak 0。
- `git fetch`：2026-07-19 后 Megatron 新 commit **15 个 <20 → 不补窗**（如实记录）。

---

## ⚠️ 等待 Rui 终审事项

1. **framing 结论与北极星口径**：Stage 1 证据显示"考题框架伪影"假设方向相反（修复侧反而更易），
   且部分由 diff 体量混杂驱动。→ **历史分数是否需重新解读、北极星口径是否修改，请 Rui 裁决**。
2. **修正口径是否升级为 v0.3 标准**：本轮修正口径仅作附表（采纳规则未触发）。是否纳入 v0.3？
3. **北极星分母去重**：261 对同一回归双计，建议 v0.3 去重为每回归一次。
4. **v2 方向**：消融指向"手册常驻在此任务上适得其反 + 宁缺毋滥过强"。后续轮次是否
   （a）弱化/移除常驻手册、（b）放宽 gating、（c）回到 v1 的 leg1 静态规则 + 检索路线并只加画像门？

---

## 诚实声明
- detector_v2 **未达 recall 判据**，如实上报冻结版；配置 Stage 3 冻结后**未用 test/dev 反馈回调**。
- 判分为 LLM（Opus），κ=1.000 复检 + 跨家族 100% + 时漂锚检 0.902；非人工真值。
- 所有资产 train-only，泄漏审计 0 外源；全程 leak_attempt=0；总花费 ~$222 ≤ $450。
- v2 的失败本身是有价值的对照：在此任务上，"给强模型灌详尽手册并令其保守"的召回崩塌，
  与 RUN2 v1 的"过度触发"构成两端，指向"精准触发而非广撒网也非过度沉默"的中间设计。

## 跑完之后
停下等 Rui。下一步（test 级终验走新前瞻窗口、单卡 GPU 微基准执行、NVIDIA 接触、v2 方向）由 Rui 决定。
