# detector_v2 — κ/跨家族/消融 (RUN3 Stage 4.2/4.3)

- generated: 2026-07-23T00:42:46.220285+00:00 · judge us.anthropic.claude-opus-4-8

## §4.2 κ 复检（100 项，pos/neg×HIT/MISS 分层，第二次判分绕缓存加 nonce）
- 样本 52 · 一致率 100.0% · **Cohen κ = 1.000** ✓ ≥0.7 继续

## §4.2 跨家族抽检（Nova Pro，50 项过采 v2 HIT）
- 样本 22 · 与 Opus-judge 一致率 **100.0%**（Opus 全链路自我偏好对照）

## §4.3 预登记消融（80 项子集：50 正+30 负，剔除冒烟重叠；Wilson 95% CI）

| 变体 | recall@2 | 95% CI | benign FPR |
|---|---|---|---|
| detector_v2 (full) | 1/50 (2.0%) | [0.4,10.5] | 0/30 (0.0%) |
| v2 − handbook | 4/50 (8.0%) | [3.2,18.8] | 0/30 (0.0%) |
| v2 − tools(父快照) | 1/50 (2.0%) | [0.4,10.5] | 0/30 (0.0%) |

- v2 vs −handbook: CI 重叠 → 不可区分
- v2 vs −tools: CI 重叠 → 不可区分

## 归因边界（§4.3 要求）
- **画像门**与**大 commit 分解**的贡献本轮**未单独隔离**（两项消融只切手册与父快照工具）。
- 画像门在 dev_tune 直接判 no_issue 的正样本数见 `reports/detector_v2_dev.md`（gated-positives 节）。
