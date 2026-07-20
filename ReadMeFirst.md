# bug_detect — 项目主目录（先读我）

MegaPerfBench：AI infra 性能回归检测器项目的全部资产。
**任何 agent 开工前必读 [`constitution.md`](constitution.md)（项目宪章）。**

## 目录地图

| 目录 | 放什么 | 一句话 |
|---|---|---|
| `dataset/` | 数据集 v0.2 发布镜像 | **唯一**允许同步 HuggingFace 的目录 |
| `detector/` | 检测器 v1 资产 | 配置/规则集/recipe map/检测器侧报告/前瞻试运行 |
| `evaluation/` | 评测设施 | harness 脚本、冻结子集、judge、baseline 成绩单、全部预测文件 |
| `paper/` | 论文素材 | claims/limitations/T1–T5 表/图数据 |
| `audit/` | 待人工判定材料 | 360 卡＋150 对抽样包（等 Rui 回填） |
| `workspace/` | 穿梭工作正本 | `ai_infra_v011_work/`（带 .git，tags v0.1–v0.2） |
| `docs/` | 项目文档源 | summary.docx 的构建源（图＋文＋脚本） |
| `archive/` | 历史归档 | 5 个传输 zip、RUN2 发货文档 |

## 全局规范

1. **每个文件夹必有 `ReadMeFirst.md`（规范）＋ `FinalReport.md`（现状概述）**，
   每次开发当次更新（宪章第 8 条）。
2. 阶段性成果后更新上级目录的 `../summary.docx`（构建源在 `docs/summary_src/`）。
3. 冻结物字节不动；修正走下一版本号（宪章 7.2）。
4. 除 `dataset/` 外一切不上公网（宪章 5.3/5.4）。
5. 与执行机的穿梭规则见宪章第 5 条；执行机上的正本路径与本地 `workspace/` 对应。

## 当前状态速览

数据集 v0.2 已冻结；检测器 v1（RUN2）已跑完并验收。整体进展、关键数字与
待办清单见 [`FinalReport.md`](FinalReport.md)。
