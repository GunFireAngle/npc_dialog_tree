"""LLM-as-judge 代理评分：对同一棵树重复评分并取均值，降低单次采样噪声。"""

from __future__ import annotations

import json
import re

from . import prompts
from .schema import Tree

KEYS = ["consistency", "persona_stability", "choice_meaningfulness"]


def _parse_scores(text: str) -> dict | None:
    s = text
    if "```" in s:
        s = s.split("```")[1]
        s = s[4:] if s.strip().startswith("json") else s
    start = s.find("{")
    if start != -1:
        depth = 0
        for i, ch in enumerate(s[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(s[start : i + 1])
                        scores = {}
                        for k in KEYS:
                            v = float(obj.get(k, 0))
                            if 1 <= v <= 5:
                                scores[k] = v
                        if len(scores) == len(KEYS):
                            scores["reason"] = str(obj.get("reason", ""))[:120]
                            return scores
                    except Exception:
                        pass
                    break
    vals = {k: re.findall(k + r"[^0-9]{0,8}([1-5])", text) for k in KEYS}
    if all(vals[k] for k in KEYS):
        out = {k: float(vals[k][0]) for k in KEYS}
        out["reason"] = "(regex-parse)"
        return out
    return None


def _as_bool(v) -> bool | None:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "是", "对", "成立", "通过", "yes", "1"):
        return True
    if s in ("false", "否", "不成立", "不通过", "no", "0"):
        return False
    return None


def _parse_checks(text: str) -> dict | None:
    """解析子项判定，并把通过率折算成三个 1-5 量表分数。"""
    s = text
    if "```" in s:
        s = s.split("```")[1]
        s = s[4:] if s.strip().startswith("json") else s
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(s[start : end + 1])
    except Exception:
        return None
    got = {k: _as_bool(obj.get(k)) for k in prompts.JUDGE_CHECKS}
    if sum(v is not None for v in got.values()) < len(prompts.JUDGE_CHECKS) - 1:
        return None
    groups: dict[str, list[int]] = {}
    for k, (scale, _) in prompts.JUDGE_CHECKS.items():
        if got[k] is None:
            continue
        groups.setdefault(scale, []).append(1 if got[k] else 0)
    out = {}
    for scale, vals in groups.items():
        out[scale] = round(1 + 4 * sum(vals) / len(vals), 3)
    if len(out) != 3:
        return None
    out["checks"] = {k: got[k] for k in prompts.JUDGE_CHECKS}
    out["reason"] = str(obj.get("reason", ""))[:120]
    return out


def judge_tree_checks(judge_engine, world: dict, npc: dict, tree: Tree, repeats: int = 3, seed: int = 0) -> dict:
    """子项判定式评审：每棵树重复 repeats 次，子项按多数票、分数按均值聚合。"""
    msgs = prompts.build_judge_check_messages(world, npc, tree.render_text())
    runs = []
    for i in range(repeats):
        raw = judge_engine.chat(msgs, max_new_tokens=220, temperature=0.3, seed=seed * 31 + i)
        sc = _parse_checks(raw)
        if sc:
            runs.append(sc)
    if not runs:
        return {**{k: float("nan") for k in KEYS}, "judge_std": float("nan"), "judge_n": 0,
                "reason": "parse_fail", "checks": {}}
    out = {k: round(sum(r[k] for r in runs) / len(runs), 3) for k in KEYS}
    per_key_std = [max(r[k] for r in runs) - min(r[k] for r in runs) for k in KEYS]
    out["judge_std"] = round(sum(per_key_std) / len(per_key_std) / 2, 3)
    out["judge_n"] = len(runs)
    out["checks"] = {
        name: round(sum(1 for r in runs if r["checks"].get(name)) / len(runs), 2)
        for name in prompts.JUDGE_CHECKS
    }
    out["reason"] = runs[0].get("reason", "")
    return out


def judge_tree(judge_engine, world: dict, npc: dict, tree: Tree, repeats: int = 3, seed: int = 0) -> dict:
    msgs = prompts.build_judge_messages(world, npc, tree.render_text())
    runs = []
    for i in range(repeats):
        raw = judge_engine.chat(msgs, max_new_tokens=200, temperature=0.3, seed=seed * 31 + i)
        sc = _parse_scores(raw)
        if sc:
            runs.append(sc)
    if not runs:
        return {**{k: float("nan") for k in KEYS}, "judge_std": float("nan"), "judge_n": 0, "reason": "parse_fail"}
    out = {k: round(sum(r[k] for r in runs) / len(runs), 3) for k in KEYS}
    per_key_std = [
        max(r[k] for r in runs) - min(r[k] for r in runs) for k in KEYS
    ]
    out["judge_std"] = round(sum(per_key_std) / len(per_key_std) / 2, 3)
    out["judge_n"] = len(runs)
    out["reason"] = runs[0].get("reason", "")
    return out
