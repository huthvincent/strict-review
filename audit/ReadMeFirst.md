# audit/ — 人工验证材料（先读我）

## 本文件夹是什么

**全项目唯一必须由人（Rui）完成的环节**：对机器标注做人工抽样验证。
回填后，数据集从"机器共识"升级为"人工校准"，直接决定论文可投档次。

## 内容清单

| 文件 | 内容 |
|---|---|
| `human_audit_packet.md` | 可读评审文件（1.2MB，510 个条目区块）：A 组 360 张卡＋B 组 150 对，逐条附证据与勾选位 |
| `human_audit_template.jsonl` | 回填模板（1,590 行 = A 组 4 字段×360 ＋ B 组 150），`verdict` 全空等填 |
| `sample_manifest.json` | 抽样清单：seed 20260719；A 组 target 357（5,016@95%/5%）实抽 360（每类≥5 补足）；B 组 A 级 100/B 级 50 按仓分层 |
| `sampling_method.md` | 抽样方法与回填后的计算口径（agreement / Cohen's κ / Wilson CI） |

## 怎么判（给 Rui）

1. 打开 `human_audit_packet.md` 顺序判：每条只判 **对 / 错 / 无法判断**
   （"无法判断"从分母剔除并单列）。预计 1–3 天。
2. 判定写回 `human_audit_template.jsonl` 的 `verdict` 字段。
3. 回填完成后交执行机跑 `apply_audits.py` → 产出 agreement/κ/CI 报告
   → 结果写进 `../paper/claims.md`（新增 C10）并更新本文件夹 FinalReport。

## 相关但不在本文件夹

前瞻 top-20 人工核验清单（NVIDIA demo 素材）在
`../detector/reports/prospective_run.md`——判法相同（真问题/误报/需查证）。

## 规范

抽样包冻结不改；如需扩样，新开 seed 新建 B 批次，不动本批。
