# dataset/ — MegaPerfBench 数据集发布镜像（先读我）

## 本文件夹是什么

数据集 **v0.2（2026-07-18 冻结）** 的发布镜像，**全项目唯一允许同步到
HuggingFace 的目录**（宪章 5.4）。正本在 `../workspace/ai_infra_v011_work/`
（git tag `v0.2`）；本目录只放对外发布需要的终版文件。

## 内容清单

| 文件 | 规模 | 说明 |
|---|---|---|
| `cards_final_v011.jsonl` | 7,737 行 | 案例卡全量（perf-related 5,016） |
| `regression_pairs_v011.jsonl` | 622 行 | 回归配对（A 级 502 / B 级 120） |
| `negatives_v011.jsonl` | 29,703 行 | 负样本（五层分层；含已知重复伪影，见下） |
| `taxonomy/taxonomy.yaml` | 74 叶 / 11 大类 | 分类学（附 raw→canonical 映射） |
| `splits/` | train/dev/test | 时间切分＋冻结协议 protocol.md＋split_stats.json＋记忆探测旁路 |
| `DATASHEET.md` | — | datasheet（其记忆探测节为 0/196 初探口径；全量 0/934 见 `../evaluation/reports/memorization.md`） |
| `dataset_stats.md` | — | v0.2 出厂检验单 |
| `README.md` | — | HuggingFace 数据卡（上传时的门面） |

## 规范

1. **本目录 = HF 同步根**。上传/更新 HF 时只从这里同步，其余目录永不上传。
2. 冻结版本字节不动。数据修正走 v0.3：在 workspace 里改 → 打新 tag → 更新本镜像
   → 同步 HF 并在 README 的 Changelog 记一行。
3. **发布纪律（宪章 5.4）**：canary 字符串（见 README 数据卡）、记录发布日期、
   建议 gated 访问。
4. 已知伪影（v0.3 待修）：`negatives_v011.jsonl` 含约 10,076 个重复 case_id；
   `splits/test.txt` 负样本行有重复、case 行含 389 张非 perf 卡。评测请使用
   `../evaluation/splits/test_eval_subset.txt`（RUN2 已去重修正：818 正＋116 对＋954 负）。
5. 消费接口 schema 见 `../workspace/ai_infra_v011_work/schemas/`。
