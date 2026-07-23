# Stage C — 最终完整性校验

- generated: 2026-07-18T06:07:04.789986+00:00
- **10/10 检查通过**

| 检查 | 结果 | 详情 |
|---|:--:|---|
| cards_final_v011 schema (0 violations) | ✅ | 7737 rows ok |
| regression_pairs_final schema | ✅ | 622 rows ok |
| s2 records schema | ✅ | 880 rows ok |
| every pair.case_id ∈ cards | ✅ | 622 ok |
| every card sha resolves in clone | ✅ | 0 unresolved |
| pair inducing+fix sha resolve (frozen set; WARN-only) | ✅ | 2 unresolved in frozen v0.1 (known limitation): vllm@5467ac319f8a->e68034999413, vllm@8d75fe48c192->3f3b6b21500b |
| every S2-joined issue_ref ∈ gh_cache | ✅ | 0 missing |
| train/dev/test pairwise disjoint | ✅ | overlaps: tr∩dv=0 tr∩te=0 dv∩te=0 |
| negatives ∩ perf-related cards == ∅ | ✅ | 0 overlap |
| frozen v0.1 files byte-identical to tag | ✅ | unchanged |

- cards 7737 · pairs 622 · negatives 29703 · s2 880
