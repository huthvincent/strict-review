# dataset/ — 现状总报告

> 截至 2026-07-19（目录整理时初立；数据内容 = v0.2，2026-07-18 冻结）。

## 状态

**v0.2 冻结完成，尚未上传 HuggingFace。** 本镜像从 workspace 正本拷出，
行数已核验：7,737 卡（perf 5,016）/ 622 对 / 29,703 负样本。

## 关键事实

- 来源：Megatron-LM / vLLM / DeepSpeed / TransformerEngine 六年 33,089 个 commit
  三层漏斗全量挖掘（Tier-1 校准闸门 recall=1.000 后放行）。
- 证据等级：622 对中 A 级 502（开发者亲述：message 明写/revert/用户 bisect），
  强于 SZZ 算法归因（最优 SZZ 对开发者亲述 oracle 仅 61% F 值）。
- 质量：Tier-2 双标 κ=0.81（kind）、taxonomy κ=0.971；记忆污染全量 0/934；
  完整性校验 10/10；基率 19.5% vs 文献 21.6%。
- novelty 定位（勿说错）：**首个 AI-infra 性能回归"引入→修复"配对数据集**；
  "首个 AI-infra 性能数据集"已被 arXiv 2506.09713 / 2512.20345 / 2506.10426 证伪。

## 已知局限（对外必须如实）

无人工真值（机器标注＋机器仲裁，人工校准包在 `../audit/` 等回填）；无 GPU 复现；
负样本重复伪影（v0.3 清理）；static_detectability 双标一致率最低（78.5%）。

## 下一步

1. Rui 决定 HF 发布形式（建议 gated）→ 按 ReadMeFirst 的发布纪律上传。
2. v0.3：清理负样本重复、pairs 个别行重复键、DATASHEET 补全量记忆探测口径。
