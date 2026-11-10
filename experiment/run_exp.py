"""主实验：世界观 × 深度 × 随机种子 × 模型（base / LoRA 微调）。

用法（项目根目录 npc_research/ 下执行）：
    python experiment/run_exp.py --models base --depths 2 --seeds 42 --limit 2   # 冒烟
    python experiment/run_exp.py                                                 # 全量
    python experiment/run_exp.py --models base,ft --judge-repeats 3              # 主实验
    python experiment/run_exp.py --ablation                                      # 附加 oneshot 消融
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment.src.worlds import WORLDS, get_world  # noqa: E402
from experiment.src.engine import get_engine  # noqa: E402
from experiment.src.builder import build_tree  # noqa: E402
from experiment.src.metrics import compute_metrics  # noqa: E402
from experiment.src.judge import judge_tree, judge_tree_checks, KEYS  # noqa: E402
from experiment.src.schema import Tree  # noqa: E402

TREE_DIR = ROOT / "experiment" / "data" / "trees"
RESULT_DIR = ROOT / "experiment" / "results"
TREES_CSV = "trees.csv"
RESULTS_CSV = "results.csv"
LOG = ROOT / "logs" / "experiment.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def tree_path(model: str, world_id: str, depth: int, seed: int, strategy: str) -> Path:
    suffix = "" if strategy == "levelwise" else f"_{strategy}"
    return TREE_DIR / model / f"{world_id}_d{depth}_s{seed}{suffix}.json"


def save_record(rec: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")


def generate_all(models, world_ids, depths, seeds, strategy, mock, gen_cfg: dict | None = None) -> None:
    for model in models:
        engine = get_engine("mock" if mock else model)
        applied = {}
        for k, v in (gen_cfg or {}).items():
            if v is not None and hasattr(engine, k):
                setattr(engine, k, v)
                applied[k] = v
        log(f"=== 生成阶段 model={model} strategy={strategy} gen={applied or '默认'} ===")
        done = 0
        for wid in world_ids:
            world = get_world(wid)
            npc = world["npc"]
            for depth in depths:
                for seed in seeds:
                    path = tree_path(model, wid, depth, seed, strategy)
                    if path.exists():
                        done += 1
                        continue
                    t0 = time.time()
                    tree = build_tree(engine, world, npc, depth, strategy, branch=2, seed=seed)
                    metrics = compute_metrics(tree, world, npc)
                    rec = {
                        "world": wid,
                        "world_name": world["name"],
                        "npc": npc["name"],
                        "method": model,
                        "depth": depth,
                        "seed": seed,
                        "strategy": strategy,
                        "gen": applied,
                        "elapsed_sec": round(time.time() - t0, 2),
                        "metrics": metrics,
                        "tree_meta": tree.meta,
                        "tree": tree.to_dict(),
                        "judge": None,
                    }
                    save_record(rec, path)
                    done += 1
                    log(
                        f"{model} {wid} d={depth} s={seed} nodes={metrics['n_nodes']}/"
                        f"{metrics['expected_nodes']} valid={metrics['structure_valid']} "
                        f"t={rec['elapsed_sec']}s"
                    )
        if hasattr(engine, "unload"):
            engine.unload()
        log(f"=== 生成阶段完成 model={model}，累计 {done} 棵 ===")


def judge_all(judge_model: str, repeats: int, mock: bool, mode: str = "checks", limit: int = 0) -> None:
    files = sorted(TREE_DIR.rglob("*.json"))
    todo = []
    for p in files:
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec.get("judge") is None:
            todo.append(p)
    if limit:
        # 分层抽样：每个 method×depth 取 2 棵，用于评审模型校准
        picked, seen = [], {}
        for p in todo:
            rec = json.loads(p.read_text(encoding="utf-8"))
            key = (rec["method"], rec["depth"])
            if seen.get(key, 0) < 2:
                seen[key] = seen.get(key, 0) + 1
                picked.append(p)
        todo = picked[:limit]
    if not todo:
        log("所有树均已有评审分数")
        return
    engine = get_engine("mock" if mock else judge_model)
    log(f"=== 评审阶段 judge={judge_model} mode={mode} 待评 {len(todo)} 棵 repeats={repeats} ===")
    for i, p in enumerate(todo, 1):
        rec = json.loads(p.read_text(encoding="utf-8"))
        world = get_world(rec["world"])
        tree = Tree.from_dict(rec["tree"])
        fn = judge_tree_checks if mode == "checks" else judge_tree
        sc = fn(engine, world, world["npc"], tree, repeats=repeats, seed=rec["seed"])
        rec["judge"] = sc
        save_record(rec, p)
        if i % 5 == 0 or i == len(todo):
            log(f"judge {i}/{len(todo)} 最新={p.name} {sc.get('consistency')}/{sc.get('persona_stability')}/{sc.get('choice_meaningfulness')}")
    if hasattr(engine, "unload"):
        engine.unload()


def recompute_metrics() -> None:
    """指标定义变化后，用已存树重算全部自动指标（不重新生成，不占显存）。"""
    files = sorted(TREE_DIR.rglob("*.json"))
    for p in files:
        rec = json.loads(p.read_text(encoding="utf-8"))
        world = get_world(rec["world"])
        tree = Tree.from_dict(rec["tree"])
        rec["metrics"] = compute_metrics(tree, world, world["npc"])
        save_record(rec, p)
    log(f"已重算 {len(files)} 棵树的自动指标")


def export_csv() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(TREE_DIR.rglob("*.json"))
    wide_cols = [
        "world", "world_name", "npc", "method", "strategy", "depth", "seed", "n_nodes",
        "expected_nodes", "completeness", "structure_valid", "depth_ok", "dup_line_rate",
        "gram_dup_rate", "distinct2", "option_dup_rate", "option_semantic_distance",
        "intent_distinct_rate", "intent_coverage", "persona_hit_rate", "persona_leak_rate",
        "meta_term_rate", "cross_world_leak_rate", "tone_drift", "mean_line_len", "std_line_len", "fact_conflicts",
        "fact_conflict_rate", "numeric_drift", "consistency", "persona_stability",
        "choice_meaningfulness", "composite",
        "judge_consistency", "judge_persona_stability", "judge_choice_meaningfulness",
        "judge_composite", "judge_n", "elapsed_sec",
    ]
    rows, long_rows = [], []
    for p in files:
        rec = json.loads(p.read_text(encoding="utf-8"))
        row = {k: rec.get(k) for k in ("world", "world_name", "npc", "method", "strategy", "depth", "seed", "elapsed_sec")}
        row.update(rec["metrics"])
        j = rec.get("judge") or {}
        for k in KEYS:
            row[f"judge_{k}"] = j.get(k)
        jv = [j.get(k) for k in KEYS]
        row["judge_composite"] = (
            round(sum(jv) / 3, 3) if all(isinstance(x, (int, float)) and x == x for x in jv) else None
        )
        row["judge_n"] = j.get("judge_n")
        rows.append(row)
        for k, v in {**rec["metrics"], **{f"judge_{kk}": j.get(kk) for kk in KEYS}}.items():
            if isinstance(v, (int, float)) and v == v:
                long_rows.append(
                    {
                        "dataset": rec["world"],
                        "method": rec["method"],
                        "params": f"depth={rec['depth']};strategy={rec['strategy']};branch=2",
                        "metric": k,
                        "value": v,
                        "elapsed_sec": rec["elapsed_sec"],
                        "seed": rec["seed"],
                    }
                )

    with open(RESULT_DIR / TREES_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=wide_cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    with open(RESULT_DIR / RESULTS_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "method", "params", "metric", "value", "elapsed_sec", "seed"])
        w.writeheader()
        w.writerows(long_rows)

    log(f"已导出 {RESULT_DIR/TREES_CSV} ({len(rows)} 行) 与 {RESULT_DIR/RESULTS_CSV} ({len(long_rows)} 行)")


def main() -> None:
    global TREE_DIR, TREES_CSV, RESULTS_CSV
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="实验标签：树写入 data/trees_<tag>，CSV 为 trees_<tag>.csv（避免覆盖主实验）")
    ap.add_argument("--models", default="base,ft")
    ap.add_argument("--worlds", default=",".join(w["id"] for w in WORLDS))
    ap.add_argument("--depths", default="2,4,6")
    ap.add_argument("--seeds", default="42,1337")
    ap.add_argument("--strategy", default="levelwise", choices=["levelwise", "oneshot", "nodewise"])
    ap.add_argument("--judge-model", default="base")
    ap.add_argument("--judge-repeats", type=int, default=3)
    ap.add_argument("--judge-mode", default="checks", choices=["checks", "holistic"])
    ap.add_argument("--judge-limit", type=int, default=0, help="仅评分层抽样的前 N 棵（校准用）")
    ap.add_argument("--limit", type=int, default=0, help="每个模型最多生成多少棵（冒烟用）")
    ap.add_argument("--skip-generate", action="store_true")
    ap.add_argument("--skip-judge", action="store_true")
    ap.add_argument("--recompute-metrics", action="store_true", help="仅用已存树重算自动指标并导出")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--temperature", type=float, default=None, help="采样温度（默认沿用引擎 0.8）")
    ap.add_argument("--top-p", type=float, default=None, help="nucleus 采样（默认 0.9）")
    ap.add_argument("--repetition-penalty", type=float, default=None, help="重复惩罚（默认 1.05）")
    ap.add_argument("--ablation", action="store_true", help="追加 oneshot 消融（仅 depth=6、前两个世界观）")
    args = ap.parse_args()

    if args.tag:
        TREE_DIR = ROOT / "experiment" / "data" / f"trees_{args.tag}"
        TREES_CSV = f"trees_{args.tag}.csv"
        RESULTS_CSV = f"results_{args.tag}.csv"

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    world_ids = [w.strip() for w in args.worlds.split(",") if w.strip()]
    depths = [int(d) for d in args.depths.split(",") if d.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    gen_cfg = {"temperature": args.temperature, "top_p": args.top_p,
               "repetition_penalty": args.repetition_penalty}

    if args.ablation:
        generate_all(models, world_ids[:2], [6], seeds[:1], "oneshot", args.mock, gen_cfg)
        export_csv()
        return

    if args.recompute_metrics:
        recompute_metrics()
        export_csv()
        return

    if args.limit:
        world_ids = world_ids[: args.limit]

    if not args.skip_generate:
        generate_all(models, world_ids, depths, seeds, args.strategy, args.mock, gen_cfg)
    if not args.skip_judge:
        judge_all(args.judge_model, args.judge_repeats, args.mock, args.judge_mode, args.judge_limit)
    export_csv()


if __name__ == "__main__":
    os.makedirs(ROOT / "logs", exist_ok=True)
    main()
