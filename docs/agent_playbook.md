# Multi-agent 作战手册（宪章第 10 条的落地）

> 主对话窗口（Rui ↔ 主 agent）是稀缺资源——context 有限、要活得久。
> 原则：**主窗口只留决策与结论，一切"重读、重算、重写"都派出去。**
> 本手册定义角色、派活模板与回报纪律。2026-07-19 初版。

## 已验证可行

本模式在 2026-07-18/19 已实战两次，效果：子 agent 共消耗 ~48 万 token 的
读取与核对工作，主窗口只收到两张结构化结论表（20 条修订建议 / 46 条验收结论）。

## 角色分工

| # | 角色 | 用什么派 | 干什么 | 回报格式 |
|---|---|---|---|---|
| 1 | **侦察员** | Explore agent（只读） | "X 文件里怎么说的 / Y 在哪 / 这两处一致吗"——一切需要翻大文件的提问 | ≤20 行：结论＋文件:行号指针＋关键引文 |
| 2 | **审计员** | Workflow 三路并行 | 穿梭回包验收（判据逐项/数据重算/跨文件一致性）；发布前体检 | verdict 表：{item, status, evidence, note} |
| 3 | **红队** | Workflow 多视角并行 | 作业书/设计稿定稿前专职找漏洞（方法学、预算、泄漏、遗漏项） | findings 表：{location, issue, severity, suggestion} |
| 4 | **工程师** | general-purpose ＋ worktree 隔离 | v2 检测器代码、数据清理脚本（`evaluation/scripts/`、dataset v0.3）——本地读写算，不碰 LLM API | 改动清单＋自测结果＋所改文件夹 FinalReport 已更新 |
| 5 | **文档员** | general-purpose | 各文件夹两件套刷新；summary.docx 重建（流程在 `docs/ReadMeFirst.md`）；HF/GitHub 同步准备 | 改动 diff 摘要＋渲染校验结论 |
| 6 | **验收员** | general-purpose → 接审计员 | 执行机回包：解压→先信 .git 核验→按内容分发→触发三路审计 | 分发清单＋审计结论表 |

## 派活模板（子 agent prompt 必含四件事）

```
1. 入职：先读 /Users/rui/Desktop/papers/AI_infra/bug_detect/constitution.md
   和 <目标文件夹>/ReadMeFirst.md。
2. 任务：<自包含描述，绝对路径，验收标准>。
3. 边界：只动 <哪些文件夹>；改动处更新该文件夹 FinalReport.md；
   不碰冻结物；花钱的事不做（记下来上报）。
4. 回报：<指定结构化格式>，不要整段粘贴大文件。
```

## 主 agent 保留事项（不外派）

与 Rui 的沟通与追问；一切拍板（架构、预算、发布）；宪章修订；
作业书与对外数字的**定稿**（子 agent 只做红队审查）；记忆维护。

## 典型战役的编排

- **v2 开发一轮**：主 agent 定设计 → 红队打设计稿 → 工程师实现（worktree）
  → 审计员核对 → 主 agent 汇总给 Rui → 文档员刷新文档。
- **穿梭一轮**：主 agent 写作业书 → 红队打作业书 → Rui 搬运 → 回包后验收员＋审计员
  → 主 agent 只读结论表做判断。
- **发布一轮**（HF/GitHub 同步）：文档员备包 → 审计员体检（禁运品检查：
  workspace/、transfer_zips/ 不得混入；canary 在位）→ 主 agent 终审 → 推送。

## 语言与上下文纪律

- 子 agent prompt 必须自包含（它看不见主对话历史）。
- 并行度：互不依赖的活一次派一批；有依赖就串行。
- 子 agent 的中间产物写进对应文件夹（磁盘是共享内存），回报只带指针。
