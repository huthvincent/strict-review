# strict-review (MegaPerfBench)

A performance-regression gatekeeper for AI-infrastructure repos (Megatron-LM,
vLLM, DeepSpeed, TransformerEngine) — built to **outperform the deployed
`/claude strict-review` bot**: statically catchable issues are flagged at
commit time with historical precedents; the rest are **routed to the repo's
existing perf CI** for measured verification.

**从这里开始 / Start here:**

1. [`constitution.md`](constitution.md) — 项目宪章。所有开发者（人或 agent）开工前必读。
2. [`ReadMeFirst.md`](ReadMeFirst.md) — 目录地图与全局规范。
3. [`FinalReport.md`](FinalReport.md) — 项目现状、关键数字、待办。

每个子目录都有自己的 `ReadMeFirst.md`（规范）与 `FinalReport.md`（现状）。

**Headline numbers so far** (frozen test subset, judge κ=0.926, leak=0):
the deployed strict-review scores 13.1% overall recall@2 with 17% FPR — a bare
frontier model matches its recall at 5% FPR; our frozen detector v1 reaches
18.6% overall at $0.233/commit, and ablations show risk-routing (leg 3)
carries nearly all recall — the v2 design follows from that.

> Note: `workspace/` (local git mirror of the executor working tree) and
> `archive/transfer_zips/` are intentionally not in this repo.
> Dataset contamination notice: labels public since **2026-07-20 UTC** —
> see [`dataset/README.md`](dataset/README.md).
