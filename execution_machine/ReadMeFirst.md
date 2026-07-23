# execution_machine/ — 执行机回运包落地区（先读我）

## 本文件夹是什么

执行机回运的结果包在本机的落地与解包区（Rui 手动放置 zip，主 agent 在此解包验收）。
当前内容：**RUN3 结果包**（`ai_infra_run3_results.zip` ＋ 解包后的 `run3_results/`，
164 文件）。

## 规范

1. 每个回运包到达后走标准验收流（见 `../docs/agent_playbook.md` 验收员＋审计员）：
   解包 → 三路验收（判据逐项/数据重算/一致性）→ 尸检/分析 → 结论落盘。
2. 包内文件**只读**——它是执行机产物的证据副本，修正意见写到验收/尸检报告里，
   修改本身回到执行机下一轮做。
3. zip 原件保留在本目录（或移入 `../archive/transfer_zips/`），作为字节级证据。
4. 验收与尸检结论的正式落点：`../detector/v2_autopsy.md`（RUN3）；
   历史轮次的验收结论在对应 FinalReport 与记忆中。

## RUN3 包内容速览

`run3_results/`：reports/（run3_report、detector_v2_dev、framing_report、
detector_v2_ablation、judge_anchor、cost_projection、gate_killrate 等 50 份）、
knowledge/（74 叶手册＋验证表＋58 微基准模板＋画像）、predictions/（520 行预测
＋judge 缓存 5,753 条）、scripts/（detector_v2 全套）、splits/、paper/、MANIFEST。
