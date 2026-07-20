# evaluation/ — 现状总报告

> 截至 2026-07-19（RUN2 验收完成）。

## 状态

评测设施完整可复用：harness（泄漏自检 3/3）、metrics（recall@budget / per-kind /
per-taxonomy / 分层 FPR+Wilson CI）、judge 已校准（双判 κ=0.926、跨家族 Nova 89%）、
考卷已冻结并开封一次。15 个预测文件逐行核验 leak_attempt=0。

## 成绩单要点（正典 = reports/baseline_results.md）

| 选手 | overall@2 | 北极星 regfix@2 | FPR |
|---|---|---|---|
| strict-review（对手） | 13.1% | 14.9% | 17% |
| 裸 Opus | 12.2% | 14.9% | **5%** |
| keyword | 0% | 0% | 5% |
| Nova Pro（跨家族） | 6.5% | 8.0% | 23% |
| **detector_v1（冻结版）** | **18.6%** | 12.3% | 14.9% |

头条发现：**手写 140 行 rubric 对召回零增益、还把误报率抬 3 倍**（strict vs 裸模型）。
Stage-2 GATE（对手 regfix 14.9% ∈ [5,45%]）通过。

## 数据完整性（2026-07-19 三路审计结论）

- 全部预测文件行数与冻结子集完全一致（1888/1334/280/61），item_id 集合零差异；
- FPR 全部独立重算吻合（164/46/44/224/142/954 及消融 46/62/13/67/400）；
- 召回数字因 judge 缓存未随包，只能验证"与数据不矛盾"（上界成立）；
- 瑕疵：baseline_megatron 有 1 行 API error＋11 个畸形 finding（不入指标）；
  xfamily 有 1 行 error。

## 待办

补运 `.judge_cache.jsonl`；paper 表格重判统一（见 `../paper/FinalReport.md`）。
