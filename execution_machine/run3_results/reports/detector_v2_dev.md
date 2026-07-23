# detector_v2 — dev 全量评测 (RUN3 Stage 4)

- generated: 2026-07-23T00:36:00.957623+00:00 · judge us.anthropic.claude-opus-4-8 (eval_judge.v1, κ 锚检 0.902) · **旧口径为主**（Stage 1 决定）
- dev_tune 250 项（150 正/pairs, 100 负）· judge 花费 ~$2.58
- 参照系（RUN2 同视图同 judge, detector_v1_dev.md）：v1 冻结 overall@2 21.3% / pair 13.3% / FPR 12.0%；v1 最强 ablate_adv overall@2 25.3% / pair 10.0% / FPR 19.0%

## 主表（旧口径，与 RUN2 同口径硬对比）

| 指标 | v2 | v1 冻结 | v1 最强(ablate_adv) | 成功判据 | 达标? |
|---|---|---|---|---|:--:|
| **overall recall@2** | **2/150 (1.3%)** | 21.3% | 25.3% | ≥25.3% | ✗ |
| **pair@2** | **0/30 (0.0%)** | 13.3% | 10.0% | ≥13.3% | ✗ |
| **weighted FPR** | **0/100 (0.0%)** | 12.0% | 19.0% | ≤15% (研究门) / ≤10% (产品线) | ✓ / ✓(产品) |

### per static_detectability recall@2 (旧口径)

| 层 | recall@2 |
|---|---|
| high | 0/40 (0.0%) |
| medium | 1/40 (2.5%) |
| low | 1/40 (2.5%) |
| pair | 0/30 (0.0%) |

## 修正口径节（附表，Stage 1 判定不作主叙事；converted 双跑）

- 转换 regfix case 数（dev_tune∩converted）: **24**（引入侧视图替换）
- **修正口径 overall@2 = 2/150 (1.3%)**（不得与 25.3% 直接比较）
- converted-case × pair 在 (repo, inducing_sha) 碰撞数: **3** → 去重敏感性 overall@2 = 2/147 (1.4%)

## 分层 benign FPR

| negative_type | FP/N | rate | Wilson95 |
|---|---|--:|---|
| false-signal-perf-infra | 0/1 | 0.0% | [0.0,79.3] |
| false-signal-smoke-ci | 0/3 | 0.0% | [0.0,56.2] |
| hard-negative-hotfile | 0/76 | 0.0% | [0.0,4.8] |
| random-benign | 0/20 | 0.0% | [-0.0,16.1] |
| **加权总计** | 0/100 | 0.0% | [0.0,3.7] |

## 画像门直接判 no_issue 的正样本

- 数量: **0**（无，✓）

## 成功判据宣判（旧口径）

- overall@2 1.3% < 25.3% → **未达标**
- pair@2 0.0% < 13.3% → **未达标**
- weighted FPR 0.0% ≤ 15% → **达标**（产品红线 10%: 达标）
- **总判定: 未全达标 — 如实报告，按消融归因，不放宽判据**

## 诚实结论与归因（Stage 4）

**v2 未达任何 recall 判据**（overall@2 1.3% ≪ 25.3%，pair 0% ≪ 13.3%），但 FPR 达标（0% ≤ 15%，亦 ≤ 产品线 10%）。

**根因（诚实报告，不粉饰）**：v2 严重**过度抑制**——250 项中仅 13 项产出 finding，其中仅 4 条 `important`、0 条 `critical`、15 条 `suggestion`。recall@2 只计 critical/important，故上限≈4。这是与 RUN2 相反的失败模式：RUN2 v1 过度触发（FPR 高），RUN3 v2 过度沉默（recall 崩）。

**消融归因（关键）**：`v2 − handbook` recall@2 **8.0%** vs 完整 v2 **2.0%**（80 项子集；Wilson CI 重叠故严格意义"不可区分"，但方向明确）。**移除手册使 recall 翻两番**，提示**常驻手册正在把模型推向"这符合某已知模式/是良性"的过度保守推理，压制了 finding**。`v2 − 父快照工具` 与 full 无差异。→ v2 的"手册常驻 + 宁缺毋滥 + 三问 gating"配方在 dev 上适得其反。

**判分可信度**：κ 复检 κ=1.000 / 一致率 100%（52 项——因 recall 极低，pos×HIT 分层仅 4 项可抽，故样本 <100，如实记录）；跨家族 Nova 一致率 100%（22 项）。判分本身高度可靠，低分不是判分问题。

**方法学纪律**：配置已在 Stage 3 冻结，第一次判分即正式成绩；**不放宽判据、不改口径、不回填**。改进（弱化/移除常驻手册、放宽"宁缺毋滥"）留待 Rui 决定的后续轮次。修正口径（引入侧）同样 1.3%，框架效应不能解释此结果（与 Stage 1 结论一致）。
