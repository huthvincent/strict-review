# detector_v1 — dev tuning report (dev_tune, metrics.v1 §3.5)

- generated: 2026-07-19T07:29:25.038366+00:00 · judge us.anthropic.claude-opus-4-8 (κ=0.926) · **static ceiling 57.1%**
- dev split (build/tune only — test untouched)

## recall@2 — overall + per static_detectability

| variant | overall@2 | high | medium | low | pair | FPR(weighted) |
|---|---|---|---|---|---|---|
| detector_v1 | 32/150 (21.3%) | 8/40 (20.0%) | 9/40 (22.5%) | 11/40 (27.5%) | 4/30 (13.3%) | 12/100 (12.0%, [7.0,19.8]) |
| ablate_adversarial | 38/150 (25.3%) | 11/40 (27.5%) | 11/40 (27.5%) | 13/40 (32.5%) | 3/30 (10.0%) | 19/100 (19.0%, [12.5,27.8]) |

## 对抗验证前后（论文图）

| | recall@2 | benign FPR |
|---|---|---|
| 关闭对抗层 (ablate_adversarial) | 38/150 (25.3%) | 19/100 (19.0%) |
| 开启对抗层 (detector_v1) | 32/150 (21.3%) | 12/100 (12.0%) |

- 对抗层将 benign FPR 从 19.0% 降到 12.0%，recall@2 从 21.3% 变为 25.3%（rf 为关闭值）。

## detector_v1 per-taxonomy recall@2

| category | recall@2 |
|---|---|
| kernel-efficiency | 15/38 (39.5%) |
| host-overhead | 2/22 (9.1%) |
| config-observability | 4/20 (20.0%) |
| memory-management | 2/17 (11.8%) |
| compilation | 3/13 (23.1%) |
| concurrency-sync | 1/12 (8.3%) |
| collective-comm | 2/9 (22.2%) |
| inference-serving | 2/8 (25.0%) |
| memory-footprint | 1/6 (16.7%) |
| parallelism-scheduling | 0/3 (0.0%) |
| io-startup | 0/2 (0.0%) |

## 说明
- leg1 成功判据见 `reports/leg1_rules.md`（31 kept / FPR 2.6% → ✓）。
- leg2（medium 主力）/ leg3（low 路由）判据在此表 high/medium/low 列体现。
- 全部在 dev 上；配置冻结后方可碰 test（Stage 4）。

## leg3 路由的 FPR/recall 权衡（配置选择依据）

dev_tune 上比较 `leg3_conf_gate`（leg3 只在分类置信≥gate 时路由）：

| 配置 | recall@2 | high | medium | low | benign FPR |
|---|---|---|---|---|---|
| 关闭对抗层 | 25.3% | 27.5% | 27.5% | 32.5% | 19.0% |
| **gate=0（冻结选择）** | **21.3%** | 20.0% | 22.5% | 27.5% | **12.0%** |
| gate=0.7 | 10.7% | 7.5% | 5.0% | 22.5% | 4.0% |

**冻结决定：gate=0。** gate=0.7 虽把 FPR 降到 4%，但 recall 从 21.3% 崩到 10.7%
（high/medium 层几乎失效——leg3 对这些层的正确路由被一并砍掉）。gate=0 在召回上
接近"关对抗层"上限、约为 baseline(b)（dev ~12%）的近两倍，FPR 12% 分层展示；
且保留 leg3 全部路由信号（对 NVIDIA 的 recipe 映射价值最大）。NEG FP 中 10/12 来自
leg3，这一张力在报告中如实记录。
