# RUN3 Stage 1 — 考题框架检验（证据整理，不下最终结论）

- generated: 2026-07-22 · **cache-only（JUDGE_CACHE_ONLY=1）**，对 test 零新模型调用
- 假设：regfix 的 case 侧（用修复 commit 提问）因方向错位被系统性压分。

> ## ★ 关键发现（方向与假设相反，务必看）
> McNemar 配对检验的方向与原假设**相反**：全部 5 个选手都是 `c > b`，即
> **「修复侧 case 命中而引入侧 pair 漏」比反向更常见**。在两个显著的选手上
> （detector_v1 p=0.0037、Megatron-strict p=0.0011），显著方向是**修复侧更易命中**，
> 不是更难。备择解释检验（§1.2）显示 case 组与 pair 组的 detectability 与 repo 分布
> **完全相同**（因为是同一批回归的 fix/inducing 两个 commit），排除了「构成差异」混杂。
> → 预登记采纳规则（≥3/5 且引入侧更易）命中 **0/5** → **不采纳修正口径为主叙事**。
> 这是**反对**「考题框架伪影压分」假设的干净证据。最终解读留给 Rui。

## 1.1 case↔pair 映射与重合率

- test pairs: 116 · fix-case 落在 test case 集内: **116 (100%)**
- **100% 重合** → McNemar 配对检验在全部 116 对上有效。

## 1.1 McNemar 配对检验（fix-侧 case vs inducing-侧 pair，同一回归）

> b = fix-case MISS 且 pair HIT（引入侧更易）· c = 反之 · p 为精确二项两侧检验

| 选手 | n | b (引入侧独中) | c (修复侧独中) | 一致命中 | 一致漏 | χ²(cc) | p(exact) | 引入侧更易? | 显著(p<.05)? |
|---|--:|--:|--:|--:|--:|--:|--:|:--:|:--:|
| detector_v1 | 116 | 6 | 22 | 4 | 84 | 8.036 | 0.0037 | ✗ | ✓ |
| baseline_megatron | 116 | 9 | 30 | 4 | 73 | 10.256 | 0.0011 | ✗ | ✓ |
| baseline_generic | 116 | 14 | 24 | 9 | 69 | 2.132 | 0.1433 | ✗ | — |
| baseline_keyword | 116 | 0 | 0 | 0 | 116 | 0.0 | 1.0 | ✗ | — |
| baseline_xfamily | 116 | 8 | 13 | 1 | 94 | 0.762 | 0.3833 | ✗ | — |

## 1.2 备择解释检验（排除『框架效应其实是构成差异』）

重合子集的 case 组（=修复 commit 的卡）与 pair 组（=引入 commit）构成对比：

- static_detectability — case: `{'low': 27, 'medium': 78, 'high': 11}` · pair: `{'low': 27, 'medium': 78, 'high': 11}`（**完全相同**）
- repo — case: `{'Megatron-LM': 11, 'vllm': 97, 'DeepSpeed': 4, 'TransformerEngine': 4}` · pair: 相同（**完全相同**）
- **diff 大小分布（dev converted 子集 n=135，字符）**：fix-case 视图 median 2342 / mean 5118；
  **inducing 视图 median 20246 / mean 17509（约 8.6× 大）**。→ 这是一个**真实的构成差异**：
  引入 commit 往往是大的功能提交，修复 commit 是小的定向补丁。方向上这**利于修复侧更易读懂**，
  与 §1.1 McNemar「修复侧更易命中」的观测一致——即观测到的方向差异**至少部分是 diff 体量差异**，
  而非纯"提问方向"效应。（详见 `reports/framing_diffsize.json`）
> 注：case 与 pair 指向同一回归的两个 commit（fix vs inducing），detectability/repo 完全相同
> （卡级标注），McNemar 的配对设计已控住这两项混杂；但 **diff 体量是未被配对消除的混杂**，须并列声明。

## 1.4 修正口径采纳规则（预登记，现在宣判）

- 规则：McNemar 在 **≥3/5** 选手上 p<0.05 **且**方向一致（引入侧更易）→ Stage 4 以修正口径为主叙事；否则旧口径为主、修正口径只作附表。
- 显著且方向正确的选手：**0/5** （[]）· 方向一致性：✗ 有反向
- **判定：旧口径为主，修正口径仅作附表**

## 1.3 引入侧视图转换率（dev regfix）

- dev regfix cases 总数 **148** · 其中 `inducing_commit_traceable ∈ {direct,likely}` **136** ·
  inducing sha 本地可解析并成功建视图 **135**（转换率 **99.3%**；1 例 sha 解析失败，0 例 git 失败）。
- converted 子集构成：detectability = `{medium:102, low:24, high:9}` · repo = `{vllm:102, Megatron-LM:26, DeepSpeed:3, TE:4}`。
- 产物 `splits/dev_regfix_inducing_views.jsonl`（135 条，PR-time 视图，leak-safe：仅白名单键，无后验字段）→ 供 Stage 4 转换双跑。

## 1.5 定位与去分母问题

- 本报告为**证据整理**：分解表 + 配对检验 + 备择解释 + 转换率（转换率见下方 1.3 步骤产物）。
- 「历史分数应如何重新解读」与「北极星口径是否修改」**留给 Rui 终审**（见 run3_report 等待项）。
- **v0.3 修正建议**：现北极星 regression-fix 分母 261 = case 侧 regfix 卡 + 116 pairs，对同一回归**双计**（fix-case 与其 pair 都计入）。建议 v0.3 去重为「每个回归计一次」。
