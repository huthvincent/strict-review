# detector/v2_prototype/ — v2 原型本地烟雾测试（先读我）

## 本文件夹是什么

v2 检测器配方（叶子手册常驻＋仓库画像＋上下文核验）的**本地小样烟雾测试**：
在 Mac 的 Claude Code 会话里用子 agent 扮演检测器与裁判，不花 Bedrock 美元。
目的：先看配方有没有真效果，再决定是否拿到执行机上大规模跑。

## 方法（宪章合规要点）

- 样本：**只用 dev 集**（宪章 1.1，test 不碰）。dev_tune 里确定性抽样
  （seed 20260719）：6 个 regression-fix 正样本（high/medium/low 各 2）＋
  4 个负样本（3 个 hard-negative-hotfile ＋ 1 个 random-benign），仅
  vllm / Megatron-LM 两仓（本地已克隆 `../../workspace/clones/`）。
- 两臂对比（同模型、同题、同判分）：**v2 原型** = diff＋手册＋画像＋
  只许查父 commit 的 git 工具＋tests 排除纪律＋budget=2；
  **bare 对照** = 只有 diff，无资料无工具。
- 判分：裁判 agent 按 eval_judge.v1 三条标准（机制吻合＋位置相关＋方向正确，
  从严）；负样本误报按 RUN2 同款确定性规则（severe 且 conf≥0.5）。
- 防泄漏：检测臂只能读 `views.json`（无任何标注字段）；`gt.json` 仅裁判可读；
  明令禁止查 commit 之后的历史。

## 文件

| 文件 | 内容 |
|---|---|
| `items.json` | 10 个样本全量（含 gt，**检测 agent 禁止读**） |
| `views.json` | 检测器可见视图（subject/files/diff/父 sha） |
| `gt.json` | 6 个正样本的标准答案（仅裁判用） |
| `handbook.md` | 迷你叶子手册（train 频次 top15 叶，确定性构建，未偷看样本） |
| `repo_profile.json` | 仓库画像 v0：两仓历史事故高发文件榜（top30，只用 train 卡） |
| `SMOKE_RESULTS.md` | 烟雾测试结果（跑完生成） |

## 与 RUN2 数字的可比性（重要）

本测试用的是**本会话的模型**（非 Bedrock Opus 4.8）、样本只有 10 个——
数字**不可**与 RUN2 正式成绩直接对比；两臂之间的对比（同模型同题）才是有效结论。
