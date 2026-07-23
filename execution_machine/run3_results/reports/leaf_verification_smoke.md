# Leaf verification 冒烟记录 (RUN3 §2.3)

- generated: 2026-07-23T00:03:01.748956+00:00 · **nvidia-smi GPU 可用: False**
- 分类分布: {'route-recipe': 15, 'microbench': 58, 'not-verifiable-yet': 1}
- 无 GPU → CPU 可验叶模板骨架冒烟 (train 卡，template-only)；GPU 全量执行属后续。

| 叶 | train 卡 id | 模板 | 冒烟状态 |
|---|---|---|---|
| algorithmic-compute-reduction | `Megatron-LM@40a4674478a5` | `knowledge/microbench_templates/algorithmic-compute-reduction.py` | repo-install-blocked, template-only smoke (parses + pytest-collectable) |
| batch-composition-mismatch | `Megatron-LM@bacd16418091` | `knowledge/microbench_templates/batch-composition-mismatch.py` | repo-install-blocked, template-only smoke (parses + pytest-collectable) |

- 冒烟叶数: 2（要求：无 GPU ≥2）· 其余 microbench 模板标 `untested`。
- **冒烟案例全部取自 train 卡**（id 见上），符合 train-only 纪律。
