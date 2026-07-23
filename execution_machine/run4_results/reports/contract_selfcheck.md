# RUN4 CONTRACT A–E 实现自查表（每条契约 → 代码行）

- 文件: `scripts/detector_v2_1.py`（除注明外）· 独立审计（general-purpose 子代理）已逐行复核确认faithful。

| 契约 | 要求 | 实现位置 | 符合 |
|---|---|---|:--:|
| **A · system prompt 原文** | §1.1 原文逐字 | `SYSTEM_V2_1`（约 44–69 行）——与 charter 行 58–84 逐字一致 | ✓ |
| A 替换① | git show/grep → read_file_at_parent/grep_at_parent | `DET_TOOLS`（工具名即 read_file_at_parent/grep_at_parent） | ✓ |
| A 替换② | "读 views3.json" → harness 递视图 | `_agentic` 组装 `umsg`（无 views3.json 字样） | ✓ |
| A 替换③ | 删 prompt 中 "confidence×0.7 并" | prompt 步骤 3 无此字样；折扣在 `_finalize` | ✓ |
| A 替换④ | 74 叶名单不进 prompt，由 handbook_lookup enum 提供 | `HANDBOOK_TOOL.input_schema.leaf.enum = _handbook_leaf_enum()`；无效叶返回提示+有效叶名 | ✓ |
| **B · route 独立腿** | touches=true ⇒ 必有 route finding；不变量恒 0 | `detect()`：`touches and not has_route` → 合成 route finding（uncategorized+conf 0.3）；`route_invariant_violation` 记 meta | ✓（冒烟不变量=0） |
| B | 生成/保留不依赖深审 | route 由分诊/首轮产出；不因深审删除（无删除路径） | ✓ |
| **C · 三问=校准器** | 模型只报 conf_raw + unverified；折扣 harness 单次施加；禁按核验降 severity/丢 finding | `_finalize`：conf_final = conf_raw×0.7 iff unverified 非空；无 severity 降级/丢弃代码 | ✓ |
| **D · 手册按需** | 不常驻；handbook_lookup(leaf) 返回页；视图附 repo_profile top 热点(≤30 行) | `HANDBOOK_TOOL` 按需；`Assets.hot_files` 取 top30 注入 `umsg` | ✓ |
| **E · 大 commit 两级审** | 廉价扫全部块→top-2 深审 5 轮；截断/跳过块记 meta；批量扫描禁单次硬塞 | `_scan_blocks`（分批+记 `scan_truncations`）；`detect` big 分支 deep_top_k=2、`skipped_blocks` 记 meta；废除 2 轮截断 | ✓ |
| **画像门落地** | 删代码短路；谓词写 meta；所有项送主审（步骤0由模型执行） | 无 `profile_gate` 短路；`meta['gate_predicate']=gate_predicate(view)`；所有项进 `_agentic` | ✓ |
| **token 记账** | 每次调用真实 usage 写 usage_ledger | `_agentic`/`_scan_blocks` 每次 `UL.record(...)`；`reports/usage_ledger.jsonl`（167 行，非全零） | ✓ |
| **统一口径 budget=2 用 conf_final** | — | `detect` 末 `findings.sort(key=conf_final)`; `[:budget]` | ✓ |

**结论**：CONTRACT A–E 全部逐条落地，仅用 §1.1 列明的四处机械替换，无其它改动。
活性门①不过是**契约内在矛盾**（见 `BLOCKERS_run4.md`），非实现偏差。
