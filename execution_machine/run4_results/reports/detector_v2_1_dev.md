# detector_v2.1 — dev 全量评测 (RUN4 v3 Stage 2)

- generated: 2026-07-23T06:47:22.947709+00:00 · judge us.anthropic.claude-opus-4-8 (eval_judge.v1, 锚检 κ=0.870) · **run4.v3 口径**
- 召回=budget2 severe 判 HIT（无 conf 门槛，metrics.v1）· FP=severe∧conf_raw≥0.5 · conf_final 仅排序
- dev_tune 250 项（150 正/pairs, 100 负）· error 0 项 (0.0%) · judge ~$3.10

## 主表（run4.v3，与 v1 同口径硬对比）

| 指标 | v2.1 | v1 冻结 | v1 最强 | 判据 | 达标 |
|---|---|---|---|---|:--:|
| overall recall@2 | **116/150 (77.3%)** | 21.3% | 25.3% | ≥25.3% | ✓ |
| pair@2 | **8/30 (26.7%)** | 13.3% | 10.0% | ≥13.3% | ✓ |
| weighted FPR | **23/100 (23.0%)** [15.8,32.2] | 12% | 19% | ≤15%/10%产品 | ✗/✗ |

## route-HIT / deep-HIT 两列（§2.3）

- route-HIT: **105** · deep-HIT: **11** （总命中 116）
- route-HIT 叶子与真值 taxonomy_label 一致: **64/105** (61%)

## 哨兵行（反博弈，§v3）

- 负样本 severe∧conf_raw<0.5 占比: **29/100 (29.0%)**
- 正样本 severe∧conf_raw<0.5 占比: 20/150 (13.3%)
- 负−正 gap = **+15.7pp** ⚠️ >15pp：**存在压 conf 躲计分嫌疑**，标注

## 分层 benign FPR (severe∧conf_raw≥0.5)

| negative_type | FP/N | rate | Wilson95 |
|---|---|--:|---|
| false-signal-perf-infra | 1/1 | 100.0% | [20.7,100.0] |
| false-signal-smoke-ci | 0/3 | 0.0% | [0.0,56.2] |
| hard-negative-hotfile | 19/76 | 25.0% | [16.6,35.8] |
| random-benign | 3/20 | 15.0% | [5.2,36.0] |
| **加权总计** | 23/100 | 23.0% | [15.8,32.2] |

## 画像门谓词 / 模型步骤0 no_issue 的正样本

- gate_predicate=true 的正样本: **0**（无）
- 模型步骤0 判 no_issue 的正样本: **0**（无）

## route 不变量
- 违规项: **0**（须=0）

## 成功判据宣判（run4.v3 口径）

- overall@2 77.3% ≥ 25.3% → **达标**
- pair@2 26.7% ≥ 13.3% → **达标**
- weighted FPR 23.0% > 15% → **未达标**（产品线10%: 未达标）
- **总判定: 未全达标 — 如实报告，不放宽不重跑**
