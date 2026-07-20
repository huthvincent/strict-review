# evaluation/ — 评测设施（先读我）

## 本文件夹是什么

整个项目的"考场"：评测协议的执行设施、冻结考卷、裁判、全部选手的答卷与成绩单。
**动这里的任何东西之前必读宪章第 1 条（方法学纪律）。**

## 内容清单

| 位置 | 内容 |
|---|---|
| `scripts/` | 代码库拷贝（正本在执行机）：eval_harness / eval_metrics / freeze_test_subset / test_harness_leakage / memorization_probe / diff_packer / mpb_common ＋ 检测器实现（detector_v1、detectors_baseline、leg1/2/3、prospective_run）＋报告生成器 |
| `splits/` | **冻结考卷**：test_eval_subset.txt（818 正＋116 对＋954 负，已去重）、test_ablation_subset.txt（1,334）、test_megatron.txt（280）、dev_calib/dev_tune、TEST_SUBSET_FROZEN 封条、memorization_flags.jsonl（934 行探测旁路） |
| `prompts/` | eval_judge.v1（判分标准，含四判例）、memorization_probe.v1、baseline_megatron.v1（对手钉版适配）、baseline_generic.v1 |
| `predictions/` | 全部答卷 JSONL：4 baseline＋detector_v1（dev+test）＋4 消融＋noMega＋prospective，每行含 finding 与 leak_attempt 字段 |
| `reports/` | baseline_results.md（**对手成绩单正典**）、judge_calibration.md（κ=0.926/Nova 89%）、memorization.md（0/934） |

## 规范

1. **test_eval_subset 已于 RUN2 Stage 4 开封一次**——后续任何评测走宪章 1.1 的三条合法路径。
2. 判分只认冻结的 `eval_judge.v1`＋Opus judge；同一批预测只判一次，下游表格全部
   从同一份判分生成（宪章 7.5——T1/T4 漂移教训）。
3. `predictions/` 是原始证据，只增不改；缺 `.judge_cache.jsonl`（在执行机，待补运）。
4. 报数规范：静态天花板 57.1% 置顶；FPR 带 Wilson 95% CI；北极星＝regression-fix
   的 recall@2（分母口径 261 = 145 case + 116 pairs，写论文时注明）。
