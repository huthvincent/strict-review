# S3 — Revert Inventory (deterministic front-end)

- generated: 2026-07-14T03:22:52.464566+00:00  |  version: `s3_reverts.v1`
- total reverts: **511**  (reapply/re-land: 17)
- with explicit 'This reverts commit <sha>': 137
- perf-motive (keyword prior, NOT confirmed): 154

## per repo

| repo | reverts |
|---|---:|
| Megatron-LM | 142 |
| vllm | 187 |
| DeepSpeed | 68 |
| TransformerEngine | 114 |

## motive-guess prior (keyword heuristic)

| motive | n |
|---|---:|
| unknown | 189 |
| perf | 154 |
| build | 84 |
| functional | 74 |
| accuracy | 10 |

## next step (needs GITHUB_TOKEN)
- Agentic pass over the ~154 perf-prior + all reapply reverts: read the reverted PR body for benchmark numbers. A reverted perf optimization → Tier-A pair (inducing = reverted commit, fix = this revert / later re-land).
- Analyst note: perf-motivated reverts are RARE; the value is the few with benchmark numbers (e.g. vLLM #33771). Most reverts are functional/build and become `not-perf`.
