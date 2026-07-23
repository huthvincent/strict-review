# RUN4 BLOCKERS — 活性门因契约内在矛盾不可过（停机等 Rui）

- 生成: 2026-07-23 · 触发点: Stage 1.2 活性门 · 依据: 纪律 #4（charter 行 33）+ §1.2（行 114）+ §5.4（行 176）
- 状态: **停机等 Rui 裁决。已完成 S0 + S1 实现与冒烟，未冻结、未进 Stage 2。**

## 一句话

按 CONTRACT A–E **逐字**实现的 detector_v2.1，活性门第①条（正样本正式计分开火 ≥8/18）
只拿到 **3/18**；经独立审计确认这是**契约本身的内在矛盾**（非实现 bug、非可通过冒烟迭代
修复），命中纪律 #4「契约本身有问题 → BLOCKERS 停机，不许变通」。

## 活性门结果（冒烟 18 正 + 7 负）

| 条件 | 结果 | 门槛 | 过? |
|---|---|---|:--:|
| ① 正样本正式计分开火 | **3/18** | ≥8/18 | ✗ |
| ② 负样本正式计分误报 | 0/7 | ≤3/7 | ✓ |
| ③ route 不变量违规 | 0 | =0 | ✓ |
| ④ leak | 0 | =0 | ✓ |

②③④ 全过，唯①差得远。实现契约的自查（每条 → 代码行）见 `reports/contract_selfcheck.md`。

## 根因：契约三处组件相乘，使"正式计分"结构上几乎不可达

1. **统一口径（charter 行 19–22）**：正式计分 = severe ∧ **conf_final ≥ 0.5**；
   `unverified` 非空 → `conf_final = conf_raw × 0.7`。
2. **CONTRACT A prompt（行 56–75）**：route finding 的 confidence **上限 0.8**（"拿不准→0.3"）；
   步骤 3 明确**邀请**模型对未核实的三问写进 `unverified`。
3. **§1.2①（行 112–113）**：活性门在 **conf_final ≥ 0.5** 上数开火。

⇒ 一条带 `unverified` 的 route finding，conf_final = conf_raw × 0.7 ≤ 0.8×0.7 = **0.56**；
要过 0.5 需 conf_raw ≥ **0.715**——而 route 本是分诊级、几乎必带 unverified。

## 数据证据（`predictions/detector_v2_1/v2_smoke.jsonl`，独立审计复核）

- **18/18 正样本在折前（conf_raw≥0.5 且 severe）全部开火**；折后只剩 **3/18**。
- 39 条 finding 中 **35 条带非空 unverified** → 全部被 ×0.7。
- 28 条 severe∧conf_raw≥0.5 → 折后仅 3 条 severe∧conf_final≥0.5。
- 幸存 3 条**全是 route**（两条 unverified 为空、一条 conf_raw 0.75）；**deep-review 贡献 0 条正式命中**。
- 正样本上 severe deep-review 的 conf_raw **最高仅 0.55** → 任何 deep-review 都过不了 0.7 折扣。
- `unverified` 理由 **48/52 为运行时固有问题**（"默认配置是否走到"、"量级多大"）——
  父快照静态视图**本就无法回答**，故必被标 unverified、必被折扣。
- 行为指标：n_turns=1 的项 18/25（模型多在首轮直接 emit，父快照工具用得少）；$/item $0.325（< $0.85 硬触发）。

## 为什么不是可迭代/可变通项

- §1.2 允许 ≤3 轮冒烟迭代，但迭代**不得改契约**（prompt 逐字、折扣 ×0.7、阈值 0.5、门控均被 §4/§5.1 冻结）。
  可调的只有"模型行为"，而模型对运行时固有问题标 unverified 是**诚实且正确**的——不该也无法靠迭代压掉。
- 纪律 #4 与 §1.2 行 114 明确：契约本身的问题 → 停机，**禁止**把门偷偷改成 conf_raw（那会一边宽一边严，违背行 20"两侧对称"）。
- 独立审计（general-purpose 子代理，逐行引证 charter）结论一致：分类 **(b) 契约内在矛盾 → BLOCKERS**，
  次要 **(c) deep-review 腿确实欠力**（正样本 severe deep conf_raw 上限 0.55）。

## 与 Mac 本地 40 题 19/24 的差异（可能的口径不一致）

Mac 原型报 19/24 命中，本轮同 prompt 只 3/18 正式计分。最可能的解释：**Mac 的计分口径与本轮
统一口径不同**（大概率 Mac 用折前 conf_raw 或不同阈值/裁判）。这正是 charter §5.1 自己警示的
"本地列不同模型/裁判/口径，禁止跨列比大小"。本轮无法取得 `detector/v2_prototype/views3.json`
（未随包传入，在 Rui 的 Mac 端）——无法直接核对 Mac 口径。

## 请 Rui 裁决（二选一或给新口径）

1. **改口径**：活性门/正式计分是否应在 **conf_raw**（折前）上计？若是，折后 conf_final 仅用于
   budget=2 排序，不用于 recall/FPR 计分门槛。（折前口径下 18/18 开火，门轻松过。）
2. **改契约**：放宽 route 的 unverified→折扣规则（例如仅 deep-review 施折扣，或 route 不因
   "运行时固有问题"标 unverified），并重新给定 CONTRACT A/统一口径。
3. 或确认维持现口径 → 则如实结论为"v2.1 在统一口径下活性门不过"，等同 v2.0 之后的第二个
   负向结果，但**根因是口径而非配方**（折前 18/18 说明配方在识别性能面上是活的）。

## 现状与已交付
- S0 完成（tag `run4-start`，锚检 96.7%/κ=0.870 通过）。
- S1 detector_v2.1 **实现完成**（CONTRACT A–E 逐条落地，自查表待补全）+ 冒烟全记录 + usage_ledger 在位。
- **未冻结、未进 Stage 2**。全程 leak=0、route 不变量=0。RUN4 至此实测花费 ~$9（含锚检/冒烟/判分）。
- 一切留档：`reports/liveness_gate.{md,json}`、`predictions/detector_v2_1/v2_smoke.jsonl`、
  `reports/usage_ledger.jsonl`、本文件。
