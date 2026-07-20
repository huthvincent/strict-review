# docs/ — 项目文档源（先读我）

## 本文件夹是什么

面向人的项目文档的**构建源**。当前唯一产物：
`/Users/rui/Desktop/papers/AI_infra/summary.docx`（30 页图文版项目全景，
通俗向，给非项目成员/未来的自己读）。

## 内容清单

| 位置 | 内容 |
|---|---|
| `summary_src/content.js` | 全文内容（DSL：h1/h2/h3/p/quote/bullets/nums/img/table/code/pb） |
| `summary_src/build.js` | docx 组装脚本（docx npm 包；解析 `**粗体**` 与 `` `等宽` ``） |
| `summary_src/make_figs.py` | 8 张配图生成（matplotlib，中文字体 Hiragino Sans GB） |
| `summary_src/figs/` | 生成的 PNG（f1 漏斗…f8 进度） |

## 更新流程（宪章 8.3）

1. 改 `content.js`（数字必须先对磁盘文件核验）；图有变则改 `make_figs.py` 并重跑。
2. 构建：`cd summary_src && python3 make_figs.py && NODE_PATH=$(npm root -g) node build.js summary.docx`
3. 校验：Mac 上用 Word AppleScript 转 PDF（`osascript` save as PDF）→ `pdftoppm`
   逐页目检（无 LibreOffice）。
4. 把新 `summary.docx` 覆盖到 `/Users/rui/Desktop/papers/AI_infra/summary.docx`。
5. 大改后建议再过一遍三路审查（数字核对 / 事实核对 / 外行读者），
   2026-07-18 首版即按此流程修掉 20 处问题。

## 规范

内容更新时同步检查 summary 与各文件夹 `FinalReport.md` 不打架——
summary 是"给人读的全景"，FinalReport 是"给 agent 读的现状"，数字必须一致。
