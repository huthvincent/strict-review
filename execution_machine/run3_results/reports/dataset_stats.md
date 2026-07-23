# MegaPerfBench — 数据集出厂检验单 (v0.2)

- generated: 2026-07-18T06:07:45.365816+00:00

## 1. 总量

- case cards: **7737** (perf-related **5016**)
- regression pairs: **622** (A=502 / B=120)
- negatives: **29703**
- S2 issue records: **880** (perf-reports 624)
- taxonomy: **74 叶 / 11 大类**

## 2. cards × kind

| kind | n |
|---|--:|
| optimization | 3591 |
| not-perf | 2611 |
| regression-fix | 816 |
| config-default-change | 487 |
| perf-infra-or-test | 150 |
| unclear | 82 |

## 3. perf cards × symptom

| symptom | n |
|---|--:|
| throughput | 1775 |
| latency | 1431 |
| memory | 756 |
| gpu-util-or-bubble | 613 |
| compile-or-startup-time | 314 |
| hang | 121 |
| n/a | 6 |

## 4. perf cards × taxonomy 大类

| category | n |
|---|--:|
| kernel-efficiency | 1556 |
| compilation | 508 |
| memory-management | 493 |
| collective-comm | 457 |
| host-overhead | 448 |
| config-observability | 430 |
| concurrency-sync | 380 |
| inference-serving | 236 |
| memory-footprint | 222 |
| io-startup | 151 |
| parallelism-scheduling | 134 |
| other | 1 |

## 5. 覆盖率与质量
- magnitude 覆盖: 300/5016 (6%)，其中 S2 issue 实测回填 69
- manifest_conditions 覆盖: 4990/5016 (99%)
- inducing 可追溯 (direct+likely): 1197
- Tier-2 双标 κ(kind): 0.81 · taxonomy 双标 κ(leaf): 0.971 · other-unclear: 0.02%
- memorization: 0/196 test 样本 memorized (0%)
- pair evidence_tier: {'A': 502, 'B': 120}
- 去重链接: 150 cards / 90 pairs

## 6. 累计成本(Phase0 v0.1 + v0.2 增量)
- v0.1 (Tier0-3+仲裁): ~$4,330
- v0.2 增量: S2 深读 ~$39 · memorization ~$1 · 去重 ~$7 · 分类学归纳+合并 ~$5 · 打标+QA ~$85 = **~$137**
- **合计 ~$4,470**（硬帽:v0.1 $6k + v0.2 作业书 $700；均未触）

## 7. 诚实声明(不变)
全机器标注 + 机器仲裁(Opus 4.8,Fable 因数据保留不可用回退)+ 机器分类,**无人类真值**。
详见 DATASHEET.md 与 reports/ARBITRATION.md。可执行复现子集与人工校准列为 future work。
