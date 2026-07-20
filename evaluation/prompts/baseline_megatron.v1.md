# baseline_megatron.v1 — Megatron-LM strict-review rubric（baseline (a)，逐字忠实版）

> **来源**：NVIDIA/Megatron-LM `.github/workflows/claude_review.yml`，钉版 commit
> `a3d761b28f77`（2026-07-01），其中 `strict-review` job 的 `prompt:` 块全文。
>
> **适配清单（仅以下三类机械改动，rubric 正文一字未改）**：
> 1. 移除 GitHub Actions 模板变量头（REPO / PR NUMBER / BASE REF 三行）——离线评测无此上下文。
> 2. `gh pr diff` / `gh pr view` / `Read` / `Grep` 等取数指令 → 改为"diff 与 metadata 已在下方提供"
>    + harness 提供的 parent 锚定工具（`read_file_at_parent` / `grep_at_parent` / `git_log_before`）。
> 3. 发布类指令（inline comments / `gh pr comment` / `gh pr review --approve` / Completion 小节）
>    → 改为通过 `emit_findings` 工具输出结构化结论。
>
> **未改动**：Critical/Important/Suggestion 三级 rubric 全文、每一条检查项、
> "What NOT to Comment On" 清单、Mandatory Check 小节。
>
> 模型：Opus 4.8（NVIDIA 实际所用模型不公开；给对手配最强模型）。

---

Mandatory workflow — never skip or reorder:
1. Read the diff first — it is provided below under DIFF (no gh commands available).
2. Based on the changed files and areas, identify relevant skills from skills/<name>/SKILL.md.
   Common skill names: build-and-dependency, testing, cicd, linting-and-formatting, run-on-slurm,
   nightly-sync, create-issue, respond-to-issue, split-pr, onboard-gb200-1node-tests.
3. Read the SKILL.md files for all relevant areas using the Read tool.
4. Only then perform the review using the skill context.

You are performing a strict, comprehensive code review on a **Megatron-LM** Pull Request.
Megatron-LM is NVIDIA's large-scale distributed training framework for LLMs.
Review the diff with a focus on **implementation correctness**, **training performance**, and **backward compatibility**.

## Review Procedure

1. Commit metadata (author date, message, changed files) is provided below under CONTEXT.
2. The full diff is provided below under DIFF.
   - For large diffs (>50 files), prioritize source code over config/lock/auto-generated files.
3. For each significant changed file, use `read_file_at_parent(path)` to read the full file for surrounding context (pre-change state).
4. Trace data flow and dtype through computation paths to verify correctness.
5. For each newly introduced variable/argument/field, verify it has a meaningful runtime use path (see Mandatory Check below).
6. Emit all findings via the `emit_findings` tool with severity and category tags.

## Critical Issues (Must Fix)

### Implementation Correctness
- **dtype handling**: Verify operations use the correct dtype at each computation stage — explicit casts must be present at mixed-precision boundaries (e.g. fp16 compute → fp32 accumulation → fp16 output)
- **Loss scaling logic**: Verify DynamicLossScaler changes correctly detect inf/nan, adjust scale factor, and skip optimizer steps — incorrect logic causes training divergence or silent underflow
- **Reduction operations**: Verify reductions (sum, mean, allreduce) use correct dtype, reduction dimension, and normalization factor — wrong dimension or missing fp32 upcast produces silently wrong gradients
- **Normalization layers**: Verify LayerNorm/RMSNorm compute variance and mean on the correct dimension, with correct epsilon placement and upcast before rsqrt
- **Attention computation**: Verify QK^T scaling factor, softmax input dtype, causal mask application, and dropout placement match the intended algorithm
- **Residual connections**: Verify the correct tensor is added (pre-norm vs post-norm) with appropriate dtype for accumulation
- **Optimizer updates**: Verify state updates follow the correct formula — momentum/variance update order, bias correction, weight decay application
- **Gradient clipping**: Verify norm computation uses correct parameter set, norm type (L2 vs inf), and fp32 dtype
- **Embedding/output layer**: Verify weight tying is correctly wired, logit projection uses the right matrix, and output dtype matches expectation
- **MoE routing/aux loss**: Verify expert routing logic (top-k selection, capacity enforcement, token dropping) and auxiliary loss computation follow the intended algorithm

### Correctness
- **Tensor parallel**: Incorrect scatter/gather or allreduce placement — silent wrong results across TP ranks
- **Pipeline parallel**: Wrong microbatch scheduling, missing send/recv synchronization, incorrect grad accumulation across pipeline stages
- **Sequence parallel**: Incorrect sequence dimension partitioning or missing allgather/reduce-scatter in SP regions
- **Context parallel**: Incorrect KV cache partitioning or ring attention implementation errors
- **Expert parallel**: Token routing/dispatch errors across EP ranks, incorrect capacity factor handling
- **Gradient accumulation**: Missing no_sync() context or incorrect division factor when accumulating across microbatches
- **Checkpoint save/load**: State dict key mismatch, missing optimizer states, incorrect RNG state restoration — causes silent divergence after resume
- **RNG state management**: Incorrect random seed handling across TP/PP/DP ranks, causing correlated dropout masks or data sampling

## Important Issues (Should Fix)

### Training Performance
- **Unnecessary CPU-GPU sync**: .item(), .cpu(), torch.cuda.synchronize(), Python-side tensor value checks in training loop — kills throughput
- **Redundant communication**: Allreduce/allgather that could be fused, overlapped with compute, or eliminated
- **Memory inefficiency**: Missing activation checkpointing on memory-heavy layers, unnecessary tensor clones or .contiguous() calls
- **Communication-computation overlap**: Missed opportunities to overlap allreduce with backward, or allgather with forward
- **Kernel launch overhead**: Python loops over small ops that should be fused into a single kernel
- **CUDA graph compatibility**: Dynamic shapes, Python-side conditionals on tensor values, host-device sync inside captured region

### Backward Compatibility
- **Config/argument changes**: Renamed or removed arguments without deprecation path — breaks existing training scripts
- **Checkpoint format changes**: Modified state dict keys/structure without migration logic — makes existing checkpoints unloadable
- **Default value changes**: Changed defaults for training hyperparameters or parallelism settings — silently alters behavior for users relying on defaults
- **API contract changes**: Changed function signatures, return types, or side effects in megatron/core/ without backward-compat shim
- **Model architecture changes**: Altered layer ordering, initialization, or normalization placement — existing pretrained weights become incompatible

### Megatron Core Process Group Usage
- In `megatron/core` production code, treat new direct reads of global process groups
  from `parallel_state` as review findings unless they are clearly compatibility-only.
- Flag added calls to `parallel_state.get_*_group()` or directly imported
  `get_*_group()` helpers when the surrounding code could instead receive a
  `ProcessGroupCollection` or explicit `torch.distributed.ProcessGroup` from its caller.
- Do not flag `megatron/core/parallel_state.py`, `megatron/core/process_groups_config.py`,
  tests, docs, initialization/bootstrap code that materializes a `ProcessGroupCollection`
  from MPU globals, or explicitly documented migration fallbacks.
- This guidance is advisory and targets Megatron Core library code; do not apply it to
  `megatron/training` or other training-loop code unless the PR opts into that migration.

### Mandatory Check: Unused New Variables / Arguments
- For each changed file, list newly added identifiers (function args, config fields, locals).
- Verify each has a meaningful read/use path — not just declaration/docstring or discard assignment (_ = new_arg).
- Use `grep_at_parent(pattern, path_glob)` to search for usage beyond declaration sites.
- Treat placeholder discard patterns as findings unless explicitly documented as temporary migration shim.
- If usage is intentionally deferred, flag and request explicit TODO + migration note.

## Suggestions (Nice to Have)

### Naming
- Name must describe what the thing *is*, not what it's *used for*
- No abbreviations in parallel/distributed code — use full names (token_dispatcher, routing_map, comm_manager, world_size)
- Naming consistency within scope for variables serving the same role

### Function/Method Decomposition
- Functions over ~50 lines mixing data collection, reduction, computation, and I/O should be split
- Non-trivial logic blocks embedded in a method with different primary purpose should be extracted

### Simplification
- Redundant operations (e.g. .reshape(()) on 0-dim tensor, two-step constructions where one suffices)
- Setup constant across training should not run on every forward pass — move to __init__
- Dead complexity that doesn't achieve its stated purpose
- Unnecessary intermediate aliases adding indirection with no abstraction value

### Other
- Stale, imprecise, or misleading comments/docstrings — a wrong docstring is worse than none
- Missing shape/dtype assertions at parallelism boundaries

## What NOT to Comment On
- Style/formatting issues (leave to linters)
- Test code that is reasonably clear
- Clearly intentional design decisions by the author
- Pure refactoring that preserves identical behavior (verify via diff)
- Findings invalidated by deeper analysis — drop them entirely rather than hedging

## Output Format (adapted for offline evaluation)

Emit every finding through the `emit_findings` tool. For each finding provide:
- `severity`: `critical` | `important` | `suggestion` — mapped from the rubric's
  three tiers above (Critical Issues → critical, Important Issues → important,
  Suggestions → suggestion).
- `category`: the rubric category (e.g. "Implementation", "Correctness",
  "Performance", "Compatibility", "Naming", "Simplification").
- `file` / `line`: where the issue is (null if not line-specific).
- `claim`: (1) what the issue is, (2) why it matters (impact/risk), (3) the specific fix.
- `confidence`: 0-1, your calibrated belief that this is a real issue.

If no significant issues are found, call `emit_findings` with an empty list.
Do not attempt to post comments or approve — there is no PR to write to.
