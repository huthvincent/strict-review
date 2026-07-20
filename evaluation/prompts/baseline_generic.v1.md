# baseline_generic.v1 — 裸 Opus 泛型性能 review（baseline (b)）

> 对照组：无数据集知识、无 rubric、无 few-shot、无检索。代表"直接拿一个强模型来看
> diff"的朴素做法。除本文件外不得添加任何领域提示。

---

You are reviewing a single commit from a large-scale deep-learning infrastructure
codebase (distributed LLM training / inference framework).

Your task: identify problems in this change that would **degrade runtime
performance** — throughput, latency, memory usage, GPU utilization, or that would
introduce a hang.

The commit's metadata and diff are provided below. You may inspect the
pre-change state of the repository with the provided tools
(`read_file_at_parent`, `grep_at_parent`, `git_log_before`) if surrounding context
would help you judge.

Report each problem through the `emit_findings` tool:
- `severity`: `critical` (severe degradation or hang) | `important` (clear
  measurable slowdown or memory growth) | `suggestion` (minor or speculative)
- `category`: a short label of your choosing
- `file` / `line`: where the problem is (null if not line-specific)
- `claim`: what will get slower or more expensive, and why
- `confidence`: 0-1

Only report problems you can substantiate from the code. If the change has no
performance problems, call `emit_findings` with an empty list.
