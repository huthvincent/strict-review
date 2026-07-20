# detector/ — 检测器资产（先读我）

## 本文件夹是什么

检测器（当前 v1，RUN2 产物）的全部资产：冻结配置、三条腿的实现产物、
检测器侧实验报告、前瞻试运行。**开发新版本前必读宪章第 1/2/3 条。**

## 内容清单

| 位置 | 内容 |
|---|---|
| `detector_v1.config.json` | 冻结配置（含 FROZEN 时间戳；字节不动） |
| `detector_v1_noMega.config.json` | 跨仓泛化变体配置 |
| `ruleset_kept.jsonl` / `ruleset_rejected.jsonl` | 腿1 静态规则：31 条留存 / 254 条被拒（负结果也是材料） |
| `ruleset_kept_noMega.jsonl` | 排除 Megatron 重挖的规则集（22 条） |
| `leg3_recipe_map.json` | **腿3 核心产物：taxonomy 叶子 → Megatron perf CI recipe 映射（可直接交付 NVIDIA）** |
| `leg2_kb_note.txt` | 腿2 检索知识库说明（评测版只用 train 3,412 卡，防泄漏） |
| `prompts/adversarial_verify.v1.md` | 对抗验证层 prompt |
| `reports/` | leg1_rules（含 noMega）、detector_v1_dev（dev 调参）、detector_v1_results（test 正式成绩单）、split_repo_holdout（跨仓）、prospective_run（**前瞻 top-20，待 Rui 人判**） |
| `prospective/unseen_commits.json` | 前瞻窗口的 61 个未见 commit 元数据 |

## 规范

1. 检测器实现代码在 `../evaluation/scripts/`（与评测共享一套代码库拷贝，正本在执行机）。
2. 新版本 = 新配置文件 + 新版本号（`detector_v2.config.json`），v1 冻结物不动。
3. **v2 的合法评测路径**见宪章 1.1——旧 test 已开封一次，禁止重复评测。
4. 接口稳定：`detect(view, tools) -> [Finding{severity, category, file, line, claim, confidence, suggested_benchmark}]`。
5. 每次改动更新本文件夹 `FinalReport.md`。
