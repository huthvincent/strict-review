# paper/ — 论文素材（先读我）

## 本文件夹是什么

论文（数据集＋检测器）的全部写作素材：主张清单、局限清单、表格、图数据。
写作规范：**每条主张必须指向仓库内具体文件与数字，并附已知反驳**（宪章 8.4）。

## 内容清单

| 位置 | 内容 |
|---|---|
| `claims.md` | 9 条可写进论文的主张（C1–C9），各带支撑文件＋已知反驳。C1 novelty 已按"首个**配对**数据集"收窄并列出三篇证伪竞品（2506.09713 / 2512.20345 / 2506.10426）；C7 引 2604.00222（ROC-AUC 0.694 弱参照）；C8 引 ICSE'21 SZZ oracle（61% F 值） |
| `limitations.md` | 9 条局限，各附缓解与未来工作 |
| `tables/` | T1 baseline 对比、T2 消融、T3 per-taxonomy、T4 per-kind、T5 分层 FPR（各有 CSV+MD） |
| `figures_data/` | 5 个作图 CSV：漏斗、taxonomy 分布、可检测性分布、对抗前后、recall@budget |

## 规范与已知警告

1. ⚠️ **T1/T4 目前不可直接用**：其 baseline 数字与
   `../evaluation/reports/baseline_results.md` 存在 2–3 个命中的重判漂移
   （T1 写 13.8/14.2，正典为 14.9；claims C4 的引用因此自相矛盾）。
   修复方式＝执行机用同一份 judge 判分重新生成（宪章 7.5）。修复前以
   baseline_results.md / detector_v1_results.md 为唯一正典。
2. 投稿路线与场所判断、人工验证惯例数字（100/357/224/487）等策略背景，
   见项目记忆与 `../../summary.docx` 第 7 章。
3. 新增素材一律注明生成脚本与数据来源文件。
