# Stage C — 去重链接报告

- generated: 2026-07-18T04:40:04.865019+00:00
- 判定模型: us.anthropic.claude-opus-4-8 (prompt dedup_judge.v1), 成本 ~$6.62

## pair 同源链接(确定性,按 inducing_sha)
- 簇数: 40 · 最大簇: 5 个 pair · 已加 `related_pairs`

## card 同源链接(Opus 判定)
- 候选(同 repo/symptom/kind + subject Jaccard≥0.5): 559
- 判为同源(conf≥0.6): 91 对 · 涉及 150 张卡加 `related_cases`
- 判定数: 559

注:同源**不删除**,仅双向链接(related_cases / related_pairs),additive。
