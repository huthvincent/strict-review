# Tier 2 — Double-Annotation Agreement

- generated: 2026-07-16T00:19:15.426290+00:00
- cards double-annotated: 1476 (20% sample)
- spend: ~$413.32

## field-level agreement

| field | agreement |
|---|---:|
| is_perf_related | 92.9% |
| kind | 87.6% |
| symptom | 82.0% |
| env_conditional | 91.0% |
| inducing_commit_traceable | 88.5% |
| static_detectability | 78.5% |

- Cohen's kappa on `kind`: **0.810**
- disagreements on `kind` queued for arbitration: 183 → `cases/audits/arbitration_queue.jsonl`

Per §6: kappa < 0.6 on a field ⇒ fix the prompt definition and re-run that field on the affected subset.
