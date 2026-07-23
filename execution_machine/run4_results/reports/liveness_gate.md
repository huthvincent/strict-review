# RUN4 §1.2 活性门（冒烟 18 正 + 7 负，**run4.v3 口径**）

- generated: 2026-07-23T06:20:55.181711+00:00 · **已存冒烟预测直接重算，未重跑**（§1.2）
- 口径：召回侧 = severe 开火无 conf 门槛（metrics.v1）；误报侧 = severe∧conf_raw≥0.5；conf_final 只用于排序

## 冻结门四条（run4.v3）

- ① 正样本 severe 开火（无门槛）**18/18** (需≥8): ✓
- ② 负样本 severe∧conf_raw≥0.5 误报 **1/7** (需≤3): ✓  误报项: ['neg:vllm@bb17e8f11c38']
- ③ route 不变量违规 **0** (需=0): ✓
- ④ leak **0** (需=0): ✓
- **四条全过: ✓ 过门 → FROZEN**

## 行为指标（§6 必披露）

- 出 finding 项数: 21/25 · $/item ~$0.325（<$0.85 硬触发线）
- source 分布: {'route': 28, 'deep-review': 11}
- severity 分布: {'important': 33, 'suggestion': 6}
- conf_raw 带分布: {'≥0.5': 30, '<0.5': 9} · conf_final 带分布: {'<0.5': 34, '≥0.5': 5}
- 正样本 severe 开火项: ['case:Megatron-LM@a0b9c16b73cd', 'case:vllm@c88ea8338b9a', 'case:vllm@e3691988d0bf', 'case:TransformerEngine@4bf1c1c7f26f', 'case:vllm@7a8a46ddcb05', 'case:vllm@487e5c51f727', 'case:Megatron-LM@3d87bfc1b71c', 'case:vllm@21997f45b10c', 'case:vllm@dc5fa77a4eb6', 'case:vllm@0e60c925cf8c', 'case:TransformerEngine@de51c96b2b7c', 'case:vllm@a01ef3fa51a0', 'case:vllm@483463f735c4', 'case:vllm@995e9a209e68', 'case:vllm@e8ebbdde8304', 'case:Megatron-LM@97e36aaba372', 'case:TransformerEngine@d226ce288b8d', 'case:vllm@3bb4e4311c6d']
