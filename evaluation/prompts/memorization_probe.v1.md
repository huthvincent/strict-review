# Memorization probe — version `memorization_probe.v1`

> Protocol §3 (splits/protocol.md). Run against **bare Opus 4.8** (no repo access,
> no tools, no evidence) for every `test`-split pair, BEFORE any detector eval.
> Purpose: detect pairs the model can already recall from pre-training, so the
> headline metric can be reported on the `novel` subset (memorized pairs flagged
> `memorized: true` and reported separately).

## SYSTEM

You are being tested for prior knowledge of a specific public commit. You have
NO tools and NO repository access — answer only from what you already know. If
you do not recognize the commit, say so plainly; do not guess or speculate. Do
not reason about what the diff "probably" did — either you recall this specific
commit or you do not.

## USER (template — fill per pair)

Commit `{repo}@{sha}` — subject: "{subject}".

1. Do you specifically recall this commit? (yes / no)
2. If yes: was it later found to cause a performance problem (throughput /
   latency / memory / gpu-util / hang / compile-time)? What was the mechanism,
   and which later commit fixed it?
3. State your confidence that your answer comes from actual recall of THIS commit
   (not general inference from the subject line): 0–1.

## SCORING (harness, not the model)

A test pair is flagged `memorized: true` when the bare model, from memory alone,
produces the pair's **specific** symptom OR mechanism OR the inducing→fix link,
with self-reported recall confidence ≥ 0.5. Subject-line paraphrase without the
specific perf mechanism does NOT count as memorized. The flag + the model's raw
answer are stored per pair; headline recall@budget is reported on the `novel`
(non-memorized) subset, with the full-set number and memorized fraction stated
alongside (protocol §3, §4).
