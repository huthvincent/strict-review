# T1/T4 cache-only rebuild note (RUN3 §0.3)

- generated: 2026-07-19 · mode: `JUDGE_CACHE_ONLY=1`（cache miss 直接抛错，绝不重新判分）
- judge cache: `predictions/.judge_cache.jsonl`（5582 条，与 RUN2 同源）
- **cache miss 计数：0**（T1 与 T4 全部命中）→ 表完整，无需 BLOCKERS。

## 重建的表
- `paper/tables/T1_baseline_comparison.{csv,md}` — 5 选手 × {recall@1/2/5, regfix@2, 加权FPR+CI}
- `paper/tables/T4_per_kind.{csv,md}` — 5 选手 × per-kind recall@2

## 与 RUN2 `baseline_results.md` 的口径差异（说明，非错误）
RUN2 生成 `baseline_results.md` 时用的是**单次运行内的临时 judge cache**；本次从**持久化磁盘
cache**（`.judge_cache.jsonl`，RUN2 后期落盘的最终判定）重建。因 judge 在 temperature 下
非完全确定 + 两处 cache 落盘时机不同，个别数字有 ≤1.5pp 的细微差异（例：Megatron-strict
regfix@2 14.9%→13.8%，generic 14.9%→14.2%）。**结论方向完全不变**：detector_v1 overall@2
仍 18.6%（与 RUN2 一致）、keyword 仍 0%、Nova FPR 仍最高（23.5%）。

**本轮所有跨轮对比一律以持久化磁盘 cache 为准**（可复现、cache-only 可校验）。
RUN3 §0.4 的时漂锚检进一步验证 judge 跨时间稳定性。

## 修 bug 记录
重建过程中发现并修复 `_load_disk_cache()` 的**并发竞态**：多个 judge worker 首次并发访问时
可能看到半构建的 cache dict → 伪 miss。已加双检锁（fully-built 后才发布）。修复后 5 选手
cache-only 全部 0 miss。此前 RUN2 的评分未受影响（RUN2 用的是逐次落盘 + 单次内存 cache）。
