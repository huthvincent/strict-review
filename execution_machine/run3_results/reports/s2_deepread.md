# Stage A — S2 深读与回填报告 (v0.1.1)

- generated: 2026-07-18T04:28:15.739244+00:00
- model: us.anthropic.claude-opus-4-8 · prompt s2_deepread.v1 · schema s2_issue.v1
- **880/880 issue 全覆盖** · 实测成本 **~$39** (帽 $120) · 截断 15 条

## report_type 分布

| type | n |
|---|--:|
| question-config | 394 |
| latent-slowness | 203 |
| not-perf | 179 |
| regression | 74 |
| measurement-artifact | 30 |

- is_perf_report: **624** / 880
- 带实测 magnitude: 491
- 有 culprit_refs: 108 · 有 fix_refs: 163
- 用户 bisect: 61

## join 与回填

- 精确 issue↔card join: **119 张卡**被链接
- magnitude 回填(issue 实测): 77 · manifest 回填: 116
- pairs_additions: cross_validated 3 + new_candidate 1
- 模糊候选(未自动 join,留后续): 437

## 产物 (v0.1.1,全部新文件,冻结 v0.1 字节未动)

- raw/issues/{vllm,Megatron-LM,DeepSpeed}.jsonl — 880 深读记录
- cases/cards_final_v011.jsonl — 7737 行(=v0.1;仅增字段)
- pairs/pairs_additions_v011.jsonl — 4
- negatives/negatives_v011.jsonl — 29,735(random-benign 深读未推翻,0 剔除)
- raw/issues/join_review.jsonl — 437(模糊候选)

注:S2 是增强流,非真值流,不做双标(符合作业书)。
