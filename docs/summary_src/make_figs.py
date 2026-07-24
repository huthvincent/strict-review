# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

plt.rcParams["font.family"] = "Hiragino Sans GB"
plt.rcParams["axes.unicode_minus"] = False

BLUE = "#3B6FB5"; BLUE_L = "#A8C4E5"
AMBER = "#D99A2B"; AMBER_L = "#F0D5A3"
GREEN = "#4C9A6E"; GREEN_L = "#BBDCC9"
RED = "#C05B5B"; RED_L = "#E8BFBF"
GRAY = "#6B7280"; GRAY_L = "#D1D5DB"
INK = "#1F2937"
OUT = "figs"

def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("saved", name)

# ---------------- F1 三层漏斗 ----------------
fig, ax = plt.subplots(figsize=(9, 4.6))
levels = [
    ("四仓全部 commit", 33089, "Megatron-LM · vLLM · DeepSpeed · TransformerEngine，六年历史", GRAY_L),
    ("Tier-1 LLM 筛查", 25759, "Tier-0 确定性路由先分流，剩下的逐个初筛（花费 $1,079）", BLUE_L),
    ("Tier-2 深度阅读", 7737, "疑似性能相关的，逐个精读打卡（花费 $2,268）", BLUE),
    ("真·性能案例卡", 5016, "确认与性能相关，每张带机制/症状/证据引文", GREEN),
    ("回归配对", 622, "其中能配成「引入 commit → 修复 commit」的对子", AMBER),
]
maxv = levels[0][1]
for i, (name, v, desc, color) in enumerate(levels):
    y = len(levels) - 1 - i
    w = max(v / maxv, 0.030)
    ax.barh(y, w, height=0.62, color=color, edgecolor="white")
    ax.text(-0.02, y, name, ha="right", va="center", fontsize=12.5, color=INK, fontweight="bold")
    ax.text(w + 0.012, y + 0.13, f"{v:,}", ha="left", va="center", fontsize=13, color=INK, fontweight="bold")
    ax.text(w + 0.012, y - 0.17, desc, ha="left", va="center", fontsize=9.5, color=GRAY)
ax.set_xlim(-0.32, 1.55); ax.set_ylim(-0.6, len(levels) - 0.4)
ax.axis("off")
ax.set_title("数据集是怎么来的：三层漏斗淘金", fontsize=15, color=INK, pad=14)
save(fig, "f1_funnel")

# ---------------- F2 数据集全景 ----------------
fig, ax = plt.subplots(figsize=(9, 4.2))
items = [
    ("负样本（没出性能问题的改动）", 29703, GRAY_L, "分五层设计，专门训练检测器「不冤枉好人」"),
    ("性能案例卡", 5016, GREEN, "深读 7,737 个 commit 后确认的真实案例"),
    ("GitHub issue 深读", 880, BLUE_L, "其中 624 份性能报告，回填实测量级进案例卡"),
    ("回归配对（引入→修复）", 622, AMBER, "A 级 502（开发者亲述）＋ B 级 120"),
    ("问题分类学（taxonomy）", 74, RED_L, "74 个叶子类别，归入 11 个大类"),
]
import math
for i, (name, v, color, desc) in enumerate(items):
    y = len(items) - 1 - i
    w = math.log10(v) / math.log10(30000)
    ax.barh(y, w, height=0.62, color=color, edgecolor="white")
    ax.text(-0.02, y, name, ha="right", va="center", fontsize=11.5, color=INK, fontweight="bold")
    ax.text(w + 0.012, y + 0.13, f"{v:,}", ha="left", va="center", fontsize=12.5, color=INK, fontweight="bold")
    ax.text(w + 0.012, y - 0.17, desc, ha="left", va="center", fontsize=9, color=GRAY)
ax.set_xlim(-0.52, 1.62); ax.set_ylim(-0.6, len(items) - 0.4)
ax.axis("off")
ax.set_title("数据集 v0.2 里都有什么（条长按对数刻度）", fontsize=15, color=INK, pad=14)
save(fig, "f2_composition")

# ---------------- F3 静态天花板 ----------------
fig, ax = plt.subplots(figsize=(9, 3.0))
segs = [("high 55", 55, BLUE), ("medium 369", 369, BLUE_L), ("n/a 43", 43, GRAY_L), ("low 351", 351, RED_L)]
total = sum(s[1] for s in segs)
x = 0
for name, v, color in segs:
    ax.barh(0, v / total, left=x, height=0.42, color=color, edgecolor="white")
    if v > 60:
        ax.text(x + v / total / 2, 0, name, ha="center", va="center", fontsize=10.5, color=INK)
    else:
        ax.text(x + v / total / 2, 0.31, name, ha="center", va="bottom", fontsize=9, color=GRAY)
    x += v / total
cut = (55 + 369 + 43) / total
ax.plot([cut, cut], [-0.38, 0.30], color=INK, lw=1.4, ls="--")
ax.annotate("57.1%：光看代码理论上能发现\n→ 交给腿 1（静态规则）＋腿 2（检索比对）",
            xy=(cut / 2, -0.44), ha="center", va="top", fontsize=10.5, color=BLUE)
ax.annotate("42.9%：看代码看不出来，必须实测\n→ 交给腿 3（路由到性能测试）",
            xy=(cut + (1 - cut) / 2, -0.44), ha="center", va="top", fontsize=10.5, color=RED)
ax.set_xlim(0, 1); ax.set_ylim(-1.05, 0.62)
ax.axis("off")
ax.set_title("test 考卷上 818 道「有问题」的题，按能否静态看出来分布", fontsize=14.5, color=INK, pad=10)
save(fig, "f3_ceiling")

# ---------------- F4 产品流水线 ----------------
fig, ax = plt.subplots(figsize=(9, 6.4))
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

def box(x, y, w, h, title, sub, fc, ec):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.10",
                       fc=fc, ec=ec, lw=1.2)
    ax.add_patch(p)
    if sub:
        ax.text(x + w / 2, y + h * 0.63, title, ha="center", va="center", fontsize=11.5, color=INK, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=9, color=GRAY)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=11.5, color=INK, fontweight="bold")

def arrow(x1, y1, x2, y2, color=GRAY):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=14, color=color, lw=1.3, shrinkA=2, shrinkB=2))

box(3.7, 9.0, 2.6, 0.85, "新 commit / PR", None, "white", INK)
arrow(5.0, 8.95, 5.0, 8.30)
box(3.2, 7.4, 3.6, 0.95, "PR-time 视图", "只含 diff＋message＋父快照", "white", INK)
arrow(4.2, 7.35, 1.9, 6.35); arrow(5.0, 7.35, 5.0, 6.35); arrow(5.8, 7.35, 8.1, 6.35)
box(0.4, 5.2, 2.9, 1.15, "腿 1 · 静态规则", "高可检类·秒级扫描", BLUE_L, BLUE)
box(3.55, 5.2, 2.9, 1.15, "腿 2 · 检索比对", "对照 5,016 张历史卡", BLUE_L, BLUE)
box(6.7, 5.2, 2.9, 1.15, "腿 3 · 风险路由", "按类别选性能测试", AMBER_L, AMBER)
arrow(1.9, 5.15, 2.9, 4.15); arrow(5.0, 5.15, 3.7, 4.15); arrow(8.15, 5.15, 8.15, 4.15)
box(1.5, 3.0, 3.6, 1.15, "对抗验证层", "专职唱反调·杀误报", BLUE_L, BLUE)
box(6.3, 3.0, 3.6, 1.15, "触发已有性能 CI", "实测吞吐·掉 10% 即 fail", AMBER_L, AMBER)
arrow(3.3, 2.95, 4.4, 1.95); arrow(8.1, 2.95, 6.4, 1.95)
box(2.9, 0.8, 4.6, 1.15, "可验证的 bug 报告", "结论＋同类历史案例＋实测数字", GREEN_L, GREEN)
ax.add_patch(mpatches.Rectangle((0.4, -0.15), 0.28, 0.28, fc=BLUE_L, ec=BLUE))
ax.text(0.8, -0.01, "本项目构建", fontsize=9.5, va="center", color=INK)
ax.add_patch(mpatches.Rectangle((2.6, -0.15), 0.28, 0.28, fc=AMBER_L, ec=AMBER))
ax.text(3.0, -0.01, "仓库已有设施（借用，不用自购 GPU）", fontsize=9.5, va="center", color=INK)
ax.set_title("最终产品：一个 commit 进来之后发生什么", fontsize=15, color=INK, pad=10)
save(fig, "f4_pipeline")

# ---------------- F5 时间切分 ----------------
fig, ax = plt.subplots(figsize=(9, 3.2))
blocks = [
    ("train 训练用", 0.0, 0.62, GREEN_L, "3,412 正 / 22,390 负 / 387 对\n（将来训模型也只准用这部分）", -0.33),
    ("dev 调参用", 0.62, 0.19, BLUE_L, "814 正 / 4,243 负 / 119 对\n（开发时反复试错的靶子）", -0.33),
    ("test 考卷", 0.81, 0.19, RED_L, "818 正 / 3,120 负 / 116 对\n（密封，全程只打开一次）", -0.72),
]
for name, x, w, color, sub, ty in blocks:
    ax.barh(0, w, left=x, height=0.34, color=color, edgecolor="white")
    ax.text(x + w / 2, 0.005, name, ha="center", va="center", fontsize=11.5, color=INK, fontweight="bold")
    ax.text(x + w / 2, ty, sub, ha="center", va="top", fontsize=9, color=GRAY)
    if ty < -0.5:
        ax.plot([x + w / 2, x + w / 2], [-0.20, ty + 0.02], color=GRAY, lw=0.8, ls=":")
ax.annotate("2026-01-01", xy=(0.62, 0.19), ha="center", fontsize=10, color=INK)
ax.annotate("2026-04-17", xy=(0.81, 0.19), ha="center", fontsize=10, color=INK)
ax.plot([0.62, 0.62], [-0.17, 0.17], color=INK, lw=1.2, ls="--")
ax.plot([0.81, 0.81], [-0.17, 0.17], color=INK, lw=1.2, ls="--")
ax.annotate("时间 →", xy=(0.985, 0.30), ha="right", fontsize=10, color=GRAY)
ax.set_xlim(0, 1); ax.set_ylim(-1.25, 0.45); ax.axis("off")
ax.set_title("按时间切分：过去当教材，最近三个月当密封考卷", fontsize=14.5, color=INK, pad=8)
save(fig, "f5_split")

# ---------------- F6 RUN2 八个 Stage ----------------
fig, ax = plt.subplots(figsize=(9, 4.8))
stages = [
    ("S1 评测地基（跑分器＋裁判）", 60, GRAY_L),
    ("S2 四个 baseline（拿到要打败的数字）", 420, BLUE_L),
    ("S3 检测器 v1 三条腿", 550, BLUE),
    ("S4 test 单次对决＋消融", 280, RED_L),
    ("S5 跨仓泛化（换个仓库还灵吗）", 120, GREEN_L),
    ("S6 前瞻试运行（真·新 commit 盲测）", 40, AMBER_L),
    ("S7 人工验证抽样包（给 Rui 判）", 20, AMBER),
    ("S8 论文素材打包", 30, GRAY_L),
]
maxv = 600
for i, (name, v, color) in enumerate(stages):
    y = len(stages) - 1 - i
    ax.barh(y, v / maxv, height=0.6, color=color, edgecolor="white")
    ax.text(-0.02, y, name, ha="right", va="center", fontsize=10.5, color=INK)
    ax.text(v / maxv + 0.012, y, f"${v}", ha="left", va="center", fontsize=10.5, color=INK, fontweight="bold")
ax.set_xlim(-1.05, 1.25); ax.set_ylim(-0.6, len(stages) - 0.4)
ax.axis("off")
ax.set_title("第二轮作业书（RUN2）：8 个 Stage 及各自分帽，估算合计 ~$1,450，总帽 $2,000", fontsize=13.5, color=INK, pad=12)
save(fig, "f6_run2")

# ---------------- F7 证据等级 ----------------
fig, ax = plt.subplots(figsize=(5.6, 3.6))
vals = [502, 120]
colors = [GREEN, BLUE_L]
wedges, _ = ax.pie(vals, colors=colors, startangle=90, counterclock=False,
                   wedgeprops=dict(width=0.42, edgecolor="white"))
ax.text(0, 0.08, "622 对", ha="center", fontsize=15, color=INK, fontweight="bold")
ax.text(0, -0.18, "回归配对", ha="center", fontsize=10.5, color=GRAY)
ax.legend(wedges, ["A 级 502 对：开发者亲述归因\n（commit message 明写 / revert / 用户 bisect）",
                   "B 级 120 对：3 票 agentic SZZ 推断"],
          loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9.5, frameon=False)
ax.set_title("回归配对的证据等级", fontsize=13.5, color=INK)
save(fig, "f7_evidence")

# ---------------- F8 项目进度条 ----------------
fig, ax = plt.subplots(figsize=(9, 4.0))
phases = [
    ("Phase 0+数据集", "校准闸门 recall=1.0\n5,016 卡/622 对冻结", GREEN, "已完成"),
    ("RUN2: v1", "v1 整体胜北极星负\n消融指路 leg3", GREEN, "已完成"),
    ("RUN3: v2.0", "哑火判负→尸检\n四处实现缺陷", GREEN, "已完成"),
    ("本地验证环", "40 题 A/B 证配方\n宪章 7.6-7.8 立", GREEN, "已完成"),
    ("RUN4: v2.1", "77.3% 召回\n工作点 0.6 定档", GREEN, "已完成"),
    ("v2.2+产品化", "误报根治/deep 强化\n前瞻滚动/接 CI", AMBER, "进行中"),
]
n = len(phases)
for i, (name, sub, color, status) in enumerate(phases):
    x = i * 1.55
    p = FancyBboxPatch((x, 1.0), 1.30, 1.5, boxstyle="round,pad=0.05,rounding_size=0.08",
                       fc=(GREEN_L if color == GREEN else (AMBER_L if color == AMBER else color)),
                       ec=GREEN if color == GREEN else (AMBER if color == AMBER else GRAY), lw=1.2)
    ax.add_patch(p)
    ax.text(x + 0.65, 2.16, name, ha="center", va="center", fontsize=10.5, color=INK, fontweight="bold")
    ax.text(x + 0.65, 1.62, sub, ha="center", va="center", fontsize=7.8, color=INK)
    ax.text(x + 0.65, 0.68, status, ha="center", va="center", fontsize=9.5,
            color=(GREEN if status == "已完成" else (AMBER if status == "进行中" else GRAY)))
    if i < n - 1:
        ax.add_patch(FancyArrowPatch((x + 1.38, 1.75), (x + 1.52, 1.75), arrowstyle="-|>",
                     mutation_scale=11, color=GRAY, lw=1.1))
ax.set_xlim(-0.2, n * 1.55); ax.set_ylim(0.3, 2.75); ax.axis("off")
ax.set_title("项目走到哪了（2026-07-22）", fontsize=14.5, color=INK, pad=10)
save(fig, "f8_roadmap")

# ---------------- F9 对决成绩 ----------------
import numpy as np
fig, ax = plt.subplots(figsize=(9, 4.6))
names = ["strict-review\n（对手）", "裸 Opus", "关键词", "Nova Pro\n（异族）", "检测器 v1\n（冻结版）"]
overall = [13.1, 12.2, 0.0, 6.5, 18.6]
regfix = [14.9, 14.9, 0.0, 8.0, 12.3]
fpr = [17.2, 4.8, 4.6, 23.5, 14.9]
x = np.arange(len(names)); w = 0.26
b1 = ax.bar(x - w, overall, w, color=BLUE, label="整体召回@2")
b2 = ax.bar(x, regfix, w, color=AMBER, label="北极星（回归修复）@2")
b3 = ax.bar(x + w, fpr, w, color=RED_L, label="误报率（越低越好）")
for bars in (b1, b2, b3):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.4, f"{b.get_height():.1f}",
                ha="center", va="bottom", fontsize=8.5, color=INK)
ax.axhline(18.0, color=GREEN, lw=1.2, ls="--")
ax.text(1.62, 19.2, "18.0 = 预登记消融（去对抗层）的北极星 —— v2 的可达性证据", fontsize=9, color=GREEN, ha="center", va="bottom")
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10)
ax.set_ylim(0, 27); ax.set_ylabel("%", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="upper left", fontsize=9, frameon=False)
ax.set_title("同一张密封考卷上的对决（RUN2 实测，判分 κ=0.926）", fontsize=14, color=INK, pad=10)
save(fig, "f9_showdown")

# ---------------- F10 检测器演进 ----------------
fig, ax = plt.subplots(figsize=(9, 4.4))
vers = ["v1\n(RUN2 冻结)", "v2.0\n(RUN3 哑火)", "v2.1 冻结口径\n(RUN4)", "v2.1 @展示门0.6\n(现役工作点)"]
overall = [21.3, 1.3, 77.3, 43.3]
pair = [13.3, 0.0, 26.7, 13.3]
fpr = [12.0, 0.0, 23.0, 2.0]
x = np.arange(len(vers)); w = 0.26
b1 = ax.bar(x - w, overall, w, color=BLUE, label="整体召回@2")
b2 = ax.bar(x, pair, w, color=AMBER, label="引入对召回@2")
b3 = ax.bar(x + w, fpr, w, color=RED_L, label="误报率（越低越好）")
for bars in (b1, b2, b3):
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.8, f"{b.get_height():.1f}",
                ha="center", va="bottom", fontsize=8.5, color=INK)
ax.axhline(10.0, color=GRAY, lw=1.0, ls=":")
ax.text(-0.42, 11.0, "产品误报红线 10%", fontsize=8.5, color=GRAY)
ax.set_xticks(x); ax.set_xticklabels(vers, fontsize=9.5)
ax.set_ylim(0, 88); ax.set_ylabel("%", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="upper left", fontsize=9, frameon=False)
ax.set_title("检测器四个版本在同一张 dev 练习卷上（RUN2/3/4 实测）", fontsize=14, color=INK, pad=10)
save(fig, "f10_evolution")
print("ALL DONE")


