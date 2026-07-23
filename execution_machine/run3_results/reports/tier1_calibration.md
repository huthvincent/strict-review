# Tier 1 — Calibration Gate Report

- generated: 2026-07-14T02:38:05.846668+00:00
- prompt_version: `tier1_screen.v1` (synthetic few-shots → leakage-free gold recall)
- gold cards: 48 | positives (perf + unclear): 29 | regression-fix: 6 | negatives: 19
- gold cards classified: 29/29 positives  ✓ all present
- candidates classified: 1226/1226

## Threshold sweep (screened-in ⟺ perf_relevance ≥ T)

| T | recall (pos) | 29-pos TP | reg-fix hit | FP (of neg) | specificity | GATE |
|---:|---:|---:|:---:|---:|---:|:---:|
| 0.05 | 1.000 | 29/29 | 6/6 | 18/19 | 0.053 | ✅ |
| 0.10 | 1.000 | 29/29 | 6/6 | 13/19 | 0.316 | ✅ |
| 0.15 | 1.000 | 29/29 | 6/6 | 12/19 | 0.368 | ✅ |
| 0.20 | 1.000 | 29/29 | 6/6 | 10/19 | 0.474 | ✅ |
| 0.25 | 1.000 | 29/29 | 6/6 | 7/19 | 0.632 | ✅ |
| 0.30 | 1.000 | 29/29 | 6/6 | 6/19 | 0.684 | ✅ |
| 0.35 | 1.000 | 29/29 | 6/6 | 6/19 | 0.684 | ✅ |
| 0.40 | 1.000 | 29/29 | 6/6 | 6/19 | 0.684 | ✅ |
| 0.45 | 0.966 | 28/29 | 6/6 | 6/19 | 0.684 | ✅ |
| 0.50 | 0.966 | 28/29 | 6/6 | 6/19 | 0.684 | ✅ |
| 0.55 | 0.931 | 27/29 | 5/6 | 6/19 | 0.684 | — |
| 0.60 | 0.793 | 23/29 | 5/6 | 4/19 | 0.789 | — |
| 0.65 | 0.793 | 23/29 | 5/6 | 3/19 | 0.842 | — |
| 0.70 | 0.793 | 23/29 | 5/6 | 3/19 | 0.842 | — |
| 0.75 | 0.655 | 19/29 | 4/6 | 2/19 | 0.895 | — |
| 0.80 | 0.621 | 18/29 | 4/6 | 2/19 | 0.895 | — |
| 0.85 | 0.552 | 16/29 | 4/6 | 0/19 | 1.000 | — |
| 0.90 | 0.241 | 7/29 | 2/6 | 0/19 | 1.000 | — |

## Gate decision

- **PASS.** The gate's job is to prove recall≥0.95 (with all 6 regression-fix hit) is ACHIEVABLE. It is: recall stays at 1.000 for every T≤0.40 and ≥0.95 up to **T=0.50**.
- **Threshold is NOT finalized here.** The full Tier-1 batch emits `perf_relevance` for all 25,759 screen commits; the operating threshold is chosen *post-batch* on the real distribution to satisfy BOTH recall≥0.95 AND the plan's ~3–5k Tier-2 target (§9). Any T in [0.35,0.40] gives full gold recall, so we have headroom to raise T if the real positive volume runs high.
- Positive rate on the 1,178 non-gold keyword candidates at T=0.50: **63.9%**. This is an UPPER bound on the screen-bucket rate (candidates are keyword-pre-filtered, so far richer than the full bucket). Real Tier-2 volume is measured from the full batch, not extrapolated from here.

## Diagnostics — positives NOT screened-in at T=0.35

- none (all positives ≥0.35). ✓

## Model's own `needs_deep_read` boolean (reference)

- recall on positives: 1.000 (29/29)
- regression-fix hit: 6/6
