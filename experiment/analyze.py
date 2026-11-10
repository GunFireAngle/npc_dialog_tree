"""统计分析与论文级图表。

用法（项目根目录 npc_research/ 下）：
    python experiment/analyze.py
产出：
    experiment/results/summary.md    均值±标准差表与 Wilcoxon 检验
    experiment/results/stats.csv     检验明细
    experiment/plots/fig1_quality_depth.{pdf,png}
    experiment/plots/fig2_structure_depth.{pdf,png}
    experiment/plots/fig3_tradeoff.{pdf,png}
    experiment/plots/fig4_ablation_oneshot.{pdf,png}（有消融数据时）
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiment" / "results"
PLOTS = ROOT / "experiment" / "plots"

JUDGE = ["consistency", "persona_stability", "choice_meaningfulness"]
AUTO = ["structure_valid", "completeness", "dup_line_rate", "distinct2",
        "option_semantic_distance", "intent_distinct_rate", "persona_hit_rate",
        "persona_leak_rate", "fact_conflicts", "mean_line_len"]

METHOD_LABEL = {"base": "Base (prompt only)", "ft": "LoRA-SFT"}
COLOR = {"base": "#4C72B0", "ft": "#C44E52"}
STYLE = {"base": "o", "ft": "s"}


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#e3e3e3",
        "grid.linewidth": 0.6,
        "grid.linestyle": "--",
        "legend.frameon": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.dpi": 150,
    })


def save(fig, name: str) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(PLOTS / f"{name}.{ext}")
    plt.close(fig)


def load() -> pd.DataFrame:
    df = pd.read_csv(RES / "trees.csv")
    df = df[df["strategy"] == "levelwise"].copy()
    df["composite"] = df[JUDGE].mean(axis=1)
    return df


def agg_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    g = df.groupby(["method", "depth"])[cols]
    out = g.agg(["mean", "std"]).round(3)
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    return out.reset_index()


def paired_test(df: pd.DataFrame, metric: str, cond_a: dict, cond_b: dict) -> tuple[str, float, float, int]:
    """按 world×seed 配对做 Wilcoxon 符号秩检验。"""
    a = df.copy()
    for k, v in cond_a.items():
        a = a[a[k] == v]
    b = df.copy()
    for k, v in cond_b.items():
        b = b[b[k] == v]
    key = ["world", "seed"]
    m = a.merge(b, on=key, suffixes=("_a", "_b"))
    x, y = m[f"{metric}_a"].values, m[f"{metric}_b"].values
    if len(x) < 5 or np.allclose(x, y):
        return "", np.nan, np.nan, len(x)
    stat, p = wilcoxon(x, y)
    delta = float(np.mean(x) - np.mean(y))
    return f"{stat:.1f}", float(p), delta, len(x)


COMPARE_METRICS = [
    "composite", "consistency", "persona_stability", "choice_meaningfulness",
    "dup_line_rate", "option_dup_rate", "intent_distinct_rate", "persona_hit_rate",
    "numeric_drift", "distinct2",
]


def load_tag(tag: str) -> pd.DataFrame | None:
    p = RES / f"trees_{tag}.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    d = d[d["strategy"] == "levelwise"].copy()
    d["composite"] = d[JUDGE].mean(axis=1)
    return d


def meta_stats(tag: str) -> dict:
    """汇总 tree_meta 里的干预计数（重采样、收尾重写、泄露）。"""
    d = ROOT / "experiment" / "data" / f"trees_{tag}"
    keys = ["calls", "dedupe_calls", "dedupe_fixed", "dedupe_gaveup",
            "reveal_retries", "beat_leak_lines", "leaf_reveal"]
    rows = []
    for p in sorted(d.glob("*/*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        tm = rec.get("tree_meta", {})
        if "beats" not in tm:
            continue
        n_leaves = max(1, len([n for n in rec["tree"]["nodes"] if not n["options"]]))
        rows.append({**{k: tm.get(k, 0) for k in keys}, "leaf_total": n_leaves})
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    out = {k: round(float(df[k].mean()), 2) for k in keys}
    out["leaf_reveal_rate"] = round(float((df["leaf_reveal"] / df["leaf_total"]).mean()), 3)
    out["trees"] = len(df)
    return out


def compare(tag: str) -> None:
    """v1（无干预）vs v2（A1+A2+A4）的探索性对照。"""
    v1, v2 = load(), load_tag(tag)
    if v2 is None:
        print(f"[compare] 找不到 {RES/f'trees_{tag}.csv'}")
        return
    keys = ["method", "world", "depth", "seed"]
    m = v1.merge(v2, on=keys, suffixes=("_v1", "_v2"))
    rows = []
    for met in COMPARE_METRICS:
        if f"{met}_v1" not in m:
            continue
        a, b = m[f"{met}_v1"].values, m[f"{met}_v2"].values
        p = np.nan
        if len(a) >= 5 and not np.allclose(a, b):
            p = float(wilcoxon(a, b).pvalue)
        rows.append({"metric": met, "v1": round(float(a.mean()), 3), "v2": round(float(b.mean()), 3),
                     "delta": round(float(b.mean() - a.mean()), 3), "n_pairs": len(a), "p": p})
    cmp_df = pd.DataFrame(rows)
    cmp_df.to_csv(RES / f"compare_{tag}.csv", index=False, encoding="utf-8-sig")

    by_depth = []
    for met in ("composite", "dup_line_rate", "option_dup_rate", "intent_distinct_rate"):
        for dep in sorted(m["depth"].unique()):
            s = m[m["depth"] == dep]
            by_depth.append({"metric": met, "depth": dep, "v1": round(float(s[f"{met}_v1"].mean()), 3),
                             "v2": round(float(s[f"{met}_v2"].mean()), 3),
                             "delta": round(float(s[f"{met}_v2"].mean() - s[f"{met}_v1"].mean()), 3)})
    depth_df = pd.DataFrame(by_depth)
    depth_df.to_csv(RES / f"compare_{tag}_by_depth.csv", index=False, encoding="utf-8-sig")

    # 图 5：干预前后的关键指标
    plot_metrics = [("composite", "Composite score"), ("dup_line_rate", "Duplicate line rate"),
                    ("option_dup_rate", "Option duplicate rate"), ("intent_distinct_rate", "Option intent distinctness")]
    fig, axes = plt.subplots(1, 4, figsize=(8.6, 2.2))
    for ax, (met, title) in zip(axes, plot_metrics):
        vals = [m[f"{met}_v1"].mean(), m[f"{met}_v2"].mean()]
        errs = [m[f"{met}_v1"].std(), m[f"{met}_v2"].std()]
        ax.bar(["v1", f"v2 (+A1/A2/A4)"], vals, color=["#8C8C8C", "#55A868"], width=0.55)
        ax.errorbar(range(2), vals, yerr=errs, fmt="none", ecolor="k", capsize=3, lw=0.8)
        ax.set_title(title, fontsize=8.5)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    save(fig, f"fig5_improvement_{tag}")

    ms = meta_stats(tag)
    lines = [f"\n\n## 干预对照 v1 vs {tag}（A1 节拍表 + A2 去重重采样 + A4 秘密门控）\n", "### 总体\n"]
    lines.append(cmp_df.round(3).to_string(index=False))
    lines.append("\n\n### 按深度\n")
    lines.append(depth_df.to_string(index=False))
    if ms:
        lines.append("\n\n### 干预执行统计（v2 每棵树均值）\n")
        lines.append("```\n" + json.dumps(ms, ensure_ascii=False, indent=2) + "\n```")
    lines.append(f"\n\n图：plots/fig5_improvement_{tag}.{{png,pdf}}（配对 n={len(m)}，探索性对照）\n")
    with open(RES / "summary.md", "a", encoding="utf-8") as f:
        f.write("".join(lines))
    print(f"[compare] -> {RES/f'compare_{tag}.csv'}, summary.md, fig5_improvement_{tag}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", default="", help="对照实验标签（如 v2），读 trees_<tag>.csv 做 v1 vs tag 对照")
    args = ap.parse_args()
    setup_style()
    df = load()
    methods = [m for m in ["base", "ft"] if m in set(df["method"])]
    depths = sorted(df["depth"].unique())

    tables = agg_table(df, JUDGE + AUTO)
    tables.to_csv(RES / "summary_agg.csv", index=False, encoding="utf-8-sig")

    # ---- 检验 ----
    rows = []
    for m in methods:
        for d1, d2 in zip(depths, depths[1:]):
            for metric in JUDGE + AUTO:
                s, p, delta, n = paired_test(df, metric, {"method": m, "depth": d1}, {"method": m, "depth": d2})
                rows.append({"method": m, "metric": metric, "test": f"depth{d1} vs depth{d2}",
                             "stat": s, "p": p, "delta": delta, "n_pairs": n})
    if len(methods) == 2:
        for d in depths:
            for metric in JUDGE + AUTO:
                s, p, delta, n = paired_test(df, metric, {"method": "base", "depth": d}, {"method": "ft", "depth": d})
                rows.append({"method": "ft-vs-base", "metric": metric, "test": f"depth{d}",
                             "stat": s, "p": p, "delta": delta, "n_pairs": n})
    stats = pd.DataFrame(rows)
    stats.to_csv(RES / "stats.csv", index=False, encoding="utf-8-sig")

    # ---- 图 1：三个量表随深度变化（确定性量表执行器） ----
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.2), sharex=True)
    titles = ["Consistency", "Persona stability", "Choice meaningfulness"]
    for ax, metric, title in zip(axes, JUDGE, titles):
        for m in methods:
            sub = df[df["method"] == m].groupby("depth")[metric]
            mu, sd = sub.mean(), sub.std().fillna(0)
            ax.errorbar(mu.index, mu.values, yerr=sd.values, marker=STYLE[m], ms=4, lw=1.2,
                        capsize=3, color=COLOR[m], label=METHOD_LABEL.get(m, m))
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Tree depth")
        ax.set_xticks(depths)
        ax.set_ylim(1, 5.2)
    axes[0].set_ylabel("Rubric score (1-5)")
    axes[0].legend(loc="lower left", fontsize=7)
    save(fig, "fig1_quality_depth")

    # ---- 图 2：结构指标随深度变化 ----
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.2), sharex=True)
    spec = [("completeness", "Completion rate"), ("dup_line_rate", "Duplicate line rate"),
            ("intent_distinct_rate", "Option intent distinctness")]
    for ax, (metric, title) in zip(axes, spec):
        for m in methods:
            sub = df[df["method"] == m].groupby("depth")[metric]
            mu, sd = sub.mean(), sub.std().fillna(0)
            ax.errorbar(mu.index, mu.values, yerr=sd.values, marker=STYLE[m], ms=4, lw=1.2,
                        capsize=3, color=COLOR[m], label=METHOD_LABEL.get(m, m))
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Tree depth")
        ax.set_xticks(depths)
    axes[0].set_ylabel("Metric value")
    axes[0].legend(loc="best", fontsize=7)
    save(fig, "fig2_structure_depth")

    # ---- 图 3：规模-质量的权衡散点 ----
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for m in methods:
        sub = df[df["method"] == m]
        ax.scatter(sub["n_nodes"], sub["composite"], s=16, alpha=0.75,
                   color=COLOR[m], marker=STYLE[m], label=METHOD_LABEL.get(m, m))
    ax.set_xscale("log")
    ax.set_xlabel("Nodes in tree (log scale)")
    ax.set_ylabel("Composite rubric score")
    ax.legend(fontsize=7)
    save(fig, "fig3_tradeoff")

    # ---- 图 4：oneshot 消融（仅 base，两策略同条件对比） ----
    all_df = pd.read_csv(RES / "trees.csv")
    ab = all_df[(all_df["strategy"] == "oneshot") & (all_df["method"] == "base")]
    if len(ab):
        lw = all_df[(all_df["strategy"] == "levelwise") & (all_df["method"] == "base") & (all_df["depth"] == 6)]
        # 只在与消融相同条件（前两个世界观、seed 42）上比较 levelwise
        lw = lw[lw["world"].isin(ab["world"].unique()) & lw["seed"].isin(ab["seed"].unique())]
        fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.3))
        for ax, metric, title in zip(axes, ["completeness", "n_nodes"], ["Completion rate", "Nodes produced (of 63)"]):
            for strat, sub, color in [("levelwise", lw, "#55A868"), ("one-shot", ab, "#8172B2")]:
                if not len(sub):
                    continue
                mu = sub[metric].mean()
                ax.bar(strat, mu, color=color, width=0.55, alpha=0.9)
                ax.errorbar(strat, mu, yerr=sub[metric].std(), color="k", capsize=3, lw=0.8)
                ax.annotate(f"{mu:.2f}", (strat, mu), textcoords="offset points",
                            xytext=(0, 14), ha="center", fontsize=7)
            ax.set_title(f"Depth 6, base: {title}", fontsize=9)
        save(fig, "fig4_ablation_oneshot")

    # ---- 评审模型校准：同一量表交给 1.5B 评审，看是否具备区分度 ----
    calib = df.dropna(subset=["judge_composite"]).copy()
    if len(calib):
        jcol = [f"judge_{k}" for k in JUDGE]
        flat = calib[jcol].nunique(dropna=True).max() == 1
        r = calib["judge_composite"].corr(calib["composite"])
        r_str = "无定义（评审分零方差）" if (flat or not np.isfinite(r)) else f"{r:.3f}"
        calib_line = (
            f"- 已评审 {len(calib)} 棵；评审分数取值个数 = {int(calib[jcol].nunique(dropna=True).max())}，"
            f"方差 = {calib[jcol].var().round(4).to_dict()}\n"
            f"- 评审合成分与确定性量表分的相关系数 = {r_str}\n"
            f"- 确定性量表分跨度 = {calib['composite'].min():.2f}~{calib['composite'].max():.2f}"
            f"（同期评审仅 {calib['judge_composite'].min():.2f}~{calib['judge_composite'].max():.2f}）\n"
        )
        if flat:
            calib_line += "- 结论：1.5B 评审对全部样本给出相同分数，不具备区分度，不能作为主评测，仅作对照证据。\n"
    else:
        calib_line = "- 尚未运行评审模型校准。\n"

    # ---- summary.md ----
    lines = ["# 实验汇总（自动生成，请人工核对后再写入论文）\n"]
    lines.append("## 评审模型校准（1.5B，9 项子检查 → 3 个量表）\n")
    lines.append(calib_line)
    for m in methods:
        lines.append(f"## {METHOD_LABEL.get(m, m)}\n")
        sub = tables[tables["method"] == m]
        cols = ["depth"] + [f"{c}_{s}" for c in JUDGE for s in ("mean", "std")]
        lines.append(sub[cols].to_string(index=False))
        lines.append("")
    lines.append("\n## 显著性检验（Wilcoxon signed-rank，配对 world×seed）\n")
    lines.append("说明：delta = mean(条件 a) − mean(条件 b)；“depth2 vs depth4”即 a=depth2、b=depth4；"
                 "“ft-vs-base”行中 a=base、b=ft（delta 为负表示 ft 更高）。\n")
    sig = stats[(stats["p"].notna()) & (stats["p"] < 0.05)]
    if len(sig):
        lines.append(sig.round(4).to_string(index=False))
    else:
        lines.append("无 p<0.05 的配对差异（样本量小或效应弱，需扩充种子数）。")
    (RES / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"done -> {RES/'summary.md'}, {RES/'stats.csv'}, plots in {PLOTS}")
    if args.compare:
        compare(args.compare)


if __name__ == "__main__":
    main()
