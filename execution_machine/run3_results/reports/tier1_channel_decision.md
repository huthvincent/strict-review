# Tier 1 channel decision — standard API, not batch

- date: 2026-07-14
- decided by: Rui (asked explicitly), executed by Claude Code

## Finding

The plan (§3, §9) assumes Tier-1 runs on the **Bedrock Batch API with Opus 4.8**
at the 5x discount (~$521–700). On this account/region that is **impossible**:

- `create-model-invocation-job` with `us.anthropic.claude-opus-4-8` (and 4.7)
  → `ValidationException: Batch inference is not supported for the requested model`.
- Same call with `us.anthropic.claude-sonnet-4-5` and `...haiku-4-5` → **accepted**.

So Bedrock Batch here supports only Sonnet/Haiku 4.5, not any Opus 4.x. (The one
pre-existing completed batch job on the account used Haiku 4.5, consistent.)

A second issue surfaced and was resolved: the `aws bedrock` CLI authenticates as
the bearer-token user `BedrockAPIKey-c0yo` when `AWS_BEARER_TOKEN_BEDROCK` is set,
and that user lacks `iam:PassRole` on `BedrockBatchInferenceRole`. Unsetting the
bearer token makes the CLI use the EC2 instance-profile role, which CAN pass the
role — but the model-support wall above is the binding constraint regardless.

## Decision

Run Tier 1 on the **standard API with Opus 4.8** (concurrent workers), preserving
the intro.md hard rule "所有 LLM 调用一律使用 claude-opus-4-8".

- Measured avg tokens/commit (from the 1,226-candidate calibration): 7,009 in /
  217 out → **$0.0405/commit** standard.
- Full screen bucket = 25,759 commits → **~$1,043** (vs the plan's ~$521 batch
  estimate). Still far under the **$6,000** hard cap; Tier1+2+3 total still lands
  in the plan's $2.5–4k band.

## Consequence for the budget line

| stage | plan estimate | revised |
|---|---:|---:|
| Tier 1 full | ~$700 (batch) | **~$1,043 (standard Opus)** |
| everything else | unchanged | unchanged (Tier 2/3 were already standard-Opus) |

Cost monitoring per §7.6 still applies: $4,500 warn line, $6,000 hard cap; each
stage's measured spend recorded in its report.

## Cleanup note

Two batch jobs were accidentally created during model-support probing (Sonnet +
Haiku, 200-record smoke input); both were immediately `Stopped` (they never left
Submitted/Validating, so ~zero inference cost) and the S3 smoke artifacts deleted.
