// MegaPerfBench summary.docx 组装脚本（解释 content.js 的 DSL）
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, BorderStyle, ShadingType,
  ImageRun, PageBreak, LevelFormat, Footer, PageNumber, VerticalAlign,
} = require("docx");

const ops = require("./content.js");
const FIGS = path.join(__dirname, "figs");

const INK = "1F2937", GRAY = "6B7280", BLUE = "3B6FB5";
const CONTENT_W = 9026; // A4, 1in margins
const FONT = { ascii: "Helvetica Neue", hAnsi: "Helvetica Neue", eastAsia: "PingFang SC" };
const MONO = { ascii: "Menlo", hAnsi: "Menlo", eastAsia: "PingFang SC" };

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}

// 解析 **bold** 与 `code` 行内标记
function runs(text, extra = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, ...extra }));
    const tok = m[0];
    if (tok.startsWith("**")) {
      out.push(new TextRun({ text: tok.slice(2, -2), bold: true, font: FONT, ...extra }));
    } else {
      out.push(new TextRun({ text: tok.slice(1, -1), font: MONO, size: (extra.size || 21) - 1, color: "374151", ...extra, bold: false }));
    }
    last = m.index + tok.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, ...extra }));
  return out;
}

const children = [];

// ---------- 封面 ----------
children.push(
  new Paragraph({ spacing: { before: 2800, after: 300 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "MegaPerfBench 项目全景梳理", bold: true, size: 64, color: INK, font: FONT })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
    children: [new TextRun({ text: "从零讲起：我们已完成的、正在做的、与最终要做成的", size: 30, color: GRAY, font: FONT })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 2000 },
    children: [new TextRun({ text: "给 Megatron-LM 等 AI 基础设施仓库造一个「更好的 /claude strict-review」", size: 24, color: GRAY, font: FONT })] }),
  new Paragraph({ alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Rui Zhu · 2026 年 7 月 22 日（第三版）", size: 24, color: INK, font: FONT })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 100 },
    children: [new TextRun({ text: "数据集 v0.2 · 四轮往返后 v2.1 现役（工作点 43.3% 召回 / 2% 误报，三轴超 v1）", size: 20, color: GRAY, font: FONT })] }),
  new Paragraph({ children: [new PageBreak()] }),
);

// ---------- 目录（手写章节列表） ----------
children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 200, after: 200 },
  children: [new TextRun({ text: "目录", bold: true, size: 32, color: INK, font: FONT })] }));
const tocItems = [
  ["阅读指南", ""],
  ["第 1 章 · 三分钟看懂整个项目", "一句话 / 一个比喻 / 全景图 / 进度"],
  ["第 2 章 · 背景：我们要解决什么问题", "AI infra / 性能回归 / 两个现役守门员"],
  ["第 3 章 · 想法的演化", "被否决的方案 → 三条腿架构"],
  ["第 4 章 · 已完成：数据集线", "漏斗 / 案例卡 / 配对 / 协议 / 质量 / 局限"],
  ["第 5 章 · 已完成：检测器线（RUN2 至 RUN4）", "v1 成绩单 / 消融 / v2 的死与生 / v2.1 工作点"],
  ["第 6 章 · 终极产品", "走一遍未来的一天 / 正面对比 / 诚实边界"],
  ["第 7 章 · 论文与竞争格局（已暂缓）", "novelty 修正 / 三个缺口 / 投稿地图"],
  ["第 8 章 · 工作方式、账本与下一步", ""],
  ["附录 A · 名词速查表", "约 60 个术语"],
  ["附录 B · 关键数字速查表", ""],
  ["附录 C · 关键文件与路径", ""],
];
for (const [t, sub] of tocItems) {
  children.push(new Paragraph({ spacing: { after: 60 }, children: [
    new TextRun({ text: t, size: 22, color: INK, font: FONT, bold: true }),
    ...(sub ? [new TextRun({ text: "　—　" + sub, size: 20, color: GRAY, font: FONT })] : []),
  ] }));
}
children.push(new Paragraph({ children: [new PageBreak()] }));

// ---------- DSL 解释 ----------
let numRefSeq = 0;
const numberingConfigs = [];

for (const op of ops) {
  if (op.h1) {
    children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 320, after: 160 },
      children: [new TextRun({ text: op.h1, bold: true, size: 32, color: INK, font: FONT })] }));
  } else if (op.h2) {
    children.push(new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 260, after: 120 },
      children: [new TextRun({ text: op.h2, bold: true, size: 26, color: INK, font: FONT })] }));
  } else if (op.h3) {
    children.push(new Paragraph({ heading: HeadingLevel.HEADING_3, spacing: { before: 200, after: 100 },
      children: [new TextRun({ text: op.h3, bold: true, size: 23, color: "374151", font: FONT })] }));
  } else if (op.p) {
    children.push(new Paragraph({ spacing: { after: 140, line: 320 }, children: runs(op.p, { size: 21, color: INK }) }));
  } else if (op.quote) {
    children.push(new Paragraph({
      spacing: { before: 80, after: 160, line: 320 }, indent: { left: 420, right: 240 },
      border: { left: { color: BLUE, style: BorderStyle.SINGLE, size: 20, space: 12 } },
      shading: { type: ShadingType.CLEAR, fill: "F4F7FB" },
      children: runs(op.quote, { size: 20, color: "374151" }),
    }));
  } else if (op.bullets) {
    for (const b of op.bullets) {
      children.push(new Paragraph({ numbering: { reference: "bullets", level: 0 },
        spacing: { after: 80, line: 300 }, children: runs(b, { size: 21, color: INK }) }));
    }
  } else if (op.nums) {
    const ref = `dec-${numRefSeq++}`;
    numberingConfigs.push({ reference: ref, levels: [{
      level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.START,
      style: { paragraph: { indent: { left: 620, hanging: 360 } } },
    }] });
    for (const n of op.nums) {
      children.push(new Paragraph({ numbering: { reference: ref, level: 0 },
        spacing: { after: 80, line: 300 }, children: runs(n, { size: 21, color: INK }) }));
    }
  } else if (op.img) {
    const file = path.join(FIGS, op.img.name + ".png");
    const { w, h } = pngSize(file);
    const targetW = Math.min(620, Math.round(w / 2));   // px at 96dpi ≈ 6.4in max
    const targetH = Math.round(targetW * h / w);
    children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 },
      children: [new ImageRun({ type: "png", data: fs.readFileSync(file),
        transformation: { width: targetW, height: targetH } })] }));
    if (op.img.cap) children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 180 },
      children: [new TextRun({ text: op.img.cap, size: 18, color: GRAY, font: FONT })] }));
  } else if (op.code) {
    for (let i = 0; i < op.code.length; i++) {
      children.push(new Paragraph({
        shading: { type: ShadingType.CLEAR, fill: "F3F4F6" },
        spacing: { after: 0, before: 0, line: 260 }, indent: { left: 300, right: 300 },
        children: [new TextRun({ text: op.code[i] || " ", font: MONO, size: 17, color: "1F2937" })],
      }));
    }
    children.push(new Paragraph({ spacing: { after: 140 }, children: [] }));
  } else if (op.table) {
    const { headers, rows } = op.table;
    const size = (op.table.size || 10) * 2;   // half-points
    const n = headers.length;
    let widths;
    if (n === 2) widths = [Math.round(CONTENT_W * 0.30), CONTENT_W - Math.round(CONTENT_W * 0.30)];
    else if (n === 3) widths = [Math.round(CONTENT_W * 0.26), Math.round(CONTENT_W * 0.20), 0].map((x, i) => i === 2 ? CONTENT_W - Math.round(CONTENT_W * 0.26) - Math.round(CONTENT_W * 0.20) : x);
    else widths = Array.from({ length: n }, () => Math.floor(CONTENT_W / n));
    const sum = widths.reduce((a, b) => a + b, 0);
    widths[n - 1] += CONTENT_W - sum;

    const mkCell = (text, isHeader, wd) => new TableCell({
      width: { size: wd, type: WidthType.DXA },
      shading: isHeader ? { type: ShadingType.CLEAR, fill: "EDF1F7" } : undefined,
      verticalAlign: VerticalAlign.CENTER,
      margins: { top: 70, bottom: 70, left: 110, right: 110 },
      children: [new Paragraph({ spacing: { after: 0, line: 270 },
        children: runs(String(text), { size, color: INK, bold: isHeader || undefined }) })],
    });
    children.push(new Table({
      width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: widths,
      rows: [
        new TableRow({ tableHeader: true, children: headers.map((hd, i) => mkCell(hd, true, widths[i])) }),
        ...rows.map(r => new TableRow({ children: r.map((c, i) => mkCell(c, false, widths[i])) })),
      ],
    }));
    children.push(new Paragraph({ spacing: { after: 160 }, children: [] }));
  } else if (op.pb) {
    children.push(new Paragraph({ children: [new PageBreak()] }));
  }
}

const doc = new Document({
  styles: { default: { document: { run: { font: FONT, size: 21, color: INK } } } },
  numbering: { config: [
    { reference: "bullets", levels: [{
      level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.START,
      style: { paragraph: { indent: { left: 620, hanging: 360 } } },
    }] },
    ...numberingConfigs,
  ] },
  sections: [{
    properties: { page: { margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, color: GRAY, font: FONT })] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then(buf => {
  const out = process.argv[2] || "summary.docx";
  fs.writeFileSync(out, buf);
  console.log("WROTE", out, buf.length, "bytes");
});
