"""对话树生成：逐层扩写（levelwise）、一次性（oneshot）、逐节点（nodewise）。"""

from __future__ import annotations

import difflib
import re
import time

from .schema import Node, Tree, parse_nodes_json, clean_text
from .metrics import intent_of
from . import prompts


def _call_with_retry(engine, msgs, seed: int, repair: bool = True, max_new_tokens: int = 1024):
    """返回 (parsed_nodes, parse_status)。status: ok / repaired / failed"""
    raw = engine.chat(msgs, max_new_tokens=max_new_tokens, seed=seed)
    try:
        return parse_nodes_json(raw), "ok"
    except Exception as e:
        if not repair:
            return [], f"failed:{e}"
        raw2 = engine.chat(
            prompts.build_repair_messages(raw, str(e)),
            max_new_tokens=max_new_tokens,
            seed=seed + 7919,
        )
        try:
            return parse_nodes_json(raw2), "repaired"
        except Exception as e2:
            return [], f"failed:{e2}"


def _fit_options(opts: list[str], branch: int) -> tuple[list[str], bool]:
    bad = len(opts) != branch
    if len(opts) > branch:
        opts = opts[:branch]
    while len(opts) < branch:
        opts.append(["……", "我明白了。"][len(opts) % 2])
    return opts, bad


DEDUPE_TRIES = 2   # A2：同一节点最多重采样次数
LINE_DUP = 0.90    # 台词与已用台词的最大相似度（只抓近乎逐字重复，避免误杀改写）
OPT_DUP = 0.90     # 选项与已用选项的最大相似度
PAIR_DUP = 0.80    # 同一节点两个选项互相的相似度

_PUNCT = re.compile(r"[\s，。！？、；：“”‘’（）《》…—\-·,.!?;:()\[\]\"']+")
_STOP = set("的了是在和与也就都不很这那你我他她它有个一把被让给对从到说会要能没还")


def _norm(s: str) -> str:
    return _PUNCT.sub("", s or "")


def _sim(a: str, b: str) -> float:
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _grams(text: str, n: int = 3) -> set[str]:
    s = _norm(text)
    return {s[i : i + n] for i in range(len(s) - n + 1) if not (set(s[i : i + n]) & _STOP)}


def _dup_reason(
    node: Node, used_lines: list[str], used_options: list[str], parent_option: str | None = None
) -> str | None:
    """A2 判定：返回 echo / line / option / pair / None。"""
    # 把玩家上一句原样当成 NPC 台词（角色串味）
    if parent_option and _sim(node.line, parent_option) >= LINE_DUP:
        return "echo"
    if used_lines and max((_sim(node.line, x) for x in used_lines[-40:]), default=0.0) >= LINE_DUP:
        return "line"
    opts = [o for o in node.options if o and o != "……"]
    for o in opts:
        if used_options and max((_sim(o, x) for x in used_options[-40:]), default=0.0) >= OPT_DUP:
            return "option"
    if len(opts) == 2 and _sim(opts[0], opts[1]) >= PAIR_DUP:
        return "pair"
    return None


def _leaks(line: str, beats: list[dict], level: int, depth: int) -> bool:
    """A4：本层台词是否提前说出了更深层才允许解锁的内容。"""
    g = _grams(line)
    for b in beats[level:]:
        if b.get("unlock") and len(g & _grams(b["unlock"])) >= 2:
            return True
    return False


def _reveals(line: str, npc: dict) -> bool:
    """A4：叶节点是否真的触及了秘密（二元组重合 ≥ 2，比三元组宽松，避免措辞改写即判失败）。"""
    return len(_grams(line, 2) & _grams(npc.get("secret", ""), 2)) >= 2


def build_levelwise(
    engine, world: dict, npc: dict, depth: int, branch: int = 2, seed: int = 0, chunk_size: int = 1
) -> Tree:
    t0 = time.time()
    tree = Tree(world["id"], npc["name"], depth, branch, "levelwise")
    beats = prompts.plan_beats(world, npc, depth, seed)
    tree.meta.update(
        {"seed": seed, "parse_ok": 0, "parse_repaired": 0, "parse_failed": 0,
         "missing_nodes": 0, "bad_branch_nodes": 0, "calls": 0,
         "dedupe_calls": 0, "dedupe_fixed": 0, "dedupe_gaveup": 0, "reveal_retries": 0,
         "beat_leak_lines": 0, "leaf_reveal": 0,
         "beats": beats, "improvements": ["A1_beat_sheet", "A2_dedupe_retry", "A4_secret_gate"]}
    )
    counter = 1
    frontier = [{"id": "n1", "depth": 1, "history": "（玩家刚走近，还没开口）",
                 "tactic": "戒备", "goal": beats[0]["goal"]}]
    counter += 1
    used_lines: list[str] = []
    used_options: list[str] = []
    call_idx = 0
    reveal_left = 8  # A4 收尾重写的整树预算

    def _call(front, level: int, blacklist: list[str] | None = None,
              force_reveal: str | None = None) -> dict:
        nonlocal call_idx
        msgs = prompts.build_expand_messages(
            world, npc, front, depth, branch, used_lines,
            blacklist=blacklist, force_reveal=force_reveal,
        )
        nodes, status = _call_with_retry(
            engine, msgs, seed * 104729 + call_idx, max_new_tokens=max(320, len(front) * 170)
        )
        call_idx += 1
        tree.meta["calls"] += 1
        if status == "ok":
            tree.meta["parse_ok"] += 1
        elif status == "repaired":
            tree.meta["parse_repaired"] += 1
        else:
            tree.meta["parse_failed"] += 1
        out: dict[str, dict] = {}
        for nd in nodes:
            out.setdefault(nd.get("id"), nd)
        # 单节点调用时，若模型回错了 id，按顺序兜底到当前节点（避免整层丢节点）
        if len(front) == 1 and nodes and front[0]["id"] not in out:
            out[front[0]["id"]] = nodes[0]
        return out

    def _make(f: dict, r: dict | None, level: int) -> Node:
        if r is None:
            tree.meta["missing_nodes"] += 1
            r = {"line": "……", "emotion": "沉默", "options": []}
        opts = [clean_text(o, 24) for o in r.get("options", []) if clean_text(o)]
        node = Node(
            id=f["id"], depth=level, line=clean_text(r.get("line", ""), 60) or "……",
            emotion=clean_text(r.get("emotion", "")) or "neutral",
            options=[], parent=f.get("parent"), parent_option=f.get("parent_option"),
        )
        if level < depth:
            opts_fit, bad = _fit_options(list(opts), branch)
            node.options = opts_fit
            tree.meta["bad_branch_nodes"] += int(bad)
        return node

    for level in range(1, depth + 1):
        next_frontier: list[dict] = []
        # 重采样预算：每层最多多花 60% 调用，避免深层把时间拖到不可接受
        dedupe_left = max(1, int(0.6 * len(frontier)))
        for f in frontier:
            res = _call([f], level)
            node = _make(f, res.get(f["id"]), level)

            # A2：与已用台词/选项/玩家上一句过近 → 带黑名单重采样
            po = f.get("parent_option")
            reason = _dup_reason(node, used_lines, used_options, po)
            initial = reason
            for _ in range(DEDUPE_TRIES):
                if not reason or dedupe_left <= 0:
                    break
                dedupe_left -= 1
                tree.meta["dedupe_calls"] += 1
                banned = used_lines[-6:] + used_options[-6:]
                if reason == "pair":
                    banned = [o for o in node.options if o]
                elif reason == "echo" and po:
                    banned = [po, node.line]
                elif node.line != "……":
                    banned = banned + [node.line]
                res2 = _call([f], level, blacklist=banned)
                if res2.get(f["id"]) is None:
                    break
                node = _make(f, res2[f["id"]], level)
                reason = _dup_reason(node, used_lines, used_options, po)
            if initial and not reason:
                tree.meta["dedupe_fixed"] += 1
            elif initial:
                tree.meta["dedupe_gaveup"] += 1

            # A4：叶节点一句都没沾到秘密时才强制重写一次，避免所有叶子收敛到同一段往事
            if (level == depth and npc.get("secret")
                    and not _reveals(node.line, npc)
                    and not (_grams(node.line, 2) & _grams(npc["secret"], 2))
                    and reveal_left > 0):
                for _ in range(1):
                    reveal_left -= 1
                    tree.meta["reveal_retries"] += 1
                    res3 = _call([f], level, blacklist=[node.line], force_reveal=npc["secret"])
                    if res3.get(f["id"]) is None:
                        break
                    node = _make(f, res3[f["id"]], level)
                    if _reveals(node.line, npc):
                        break

            # A4：提前泄露 / 叶子是否真揭示（只统计，不改写）
            if level < depth and _leaks(node.line, beats, level, depth):
                tree.meta["beat_leak_lines"] += 1
            if level == depth and _reveals(node.line, npc):
                tree.meta["leaf_reveal"] += 1

            tree.add(node)
            used_lines.append(node.line)
            used_options.extend(node.options)
            if level < depth:
                for i, opt in enumerate(node.options):
                    nxt = level + 1
                    tactic = prompts.plan_tactic(nxt, depth, intent_of(opt), f"{node.id}-{i}")
                    goal = beats[nxt - 1]["goal"]
                    leaf_mode = None
                    if nxt >= depth:
                        # 叶子统一走“兑现”（A5 双收尾探针无正收益，已回滚）
                        leaf_mode = "reveal"
                        tactic = "揭示"
                    next_frontier.append(
                        {
                            "id": f"n{counter}",
                            "depth": nxt,
                            "parent": node.id,
                            "parent_option": opt,
                            "tactic": tactic,
                            "leaf_mode": leaf_mode,
                            "history": f"{f['history']} ｜ NPC：{node.line} ｜ 玩家：{opt}",
                            "goal": goal,
                        }
                    )
                    counter += 1
        frontier = next_frontier

    tree.meta["elapsed"] = round(time.time() - t0, 2)
    tree.meta["strategy_calls"] = tree.meta["calls"]
    return tree


def build_oneshot(engine, world: dict, npc: dict, depth: int, branch: int = 2, seed: int = 0) -> Tree:
    t0 = time.time()
    tree = Tree(world["id"], npc["name"], depth, branch, "oneshot")
    tree.meta.update({"seed": seed, "parse_ok": 0, "parse_repaired": 0, "parse_failed": 1,
                      "missing_nodes": 0, "bad_branch_nodes": 0, "calls": 1})
    msgs = prompts.build_oneshot_messages(world, npc, depth, branch)
    n_expected = sum(branch ** i for i in range(depth))
    nodes, status = _call_with_retry(
        engine, msgs, seed * 104729, max_new_tokens=min(7168, max(600, n_expected * 130))
    )
    if status != "failed":
        tree.meta["parse_ok"] = int(status == "ok")
        tree.meta["parse_repaired"] = int(status == "repaired")
        tree.meta["parse_failed"] = 0
    # 按层序编号还原父子关系
    expected = [sum(branch ** i for i in range(d)) for d in range(1, depth + 1)]
    ids = [f"n{i + 1}" for i in range(len(nodes))]
    for idx, nd in enumerate(nodes):
        d = 1
        for lv, cum in enumerate(expected, start=1):
            if idx + 1 <= cum:
                d = lv
                break
        node = Node(
            id=ids[idx], depth=d, line=nd["line"], emotion=nd["emotion"],
            options=nd["options"] if d < depth else [], parent=None, parent_option=None,
        )
        tree.add(node)
    for idx in range(1, len(nodes)):
        parent_idx = (idx - 1) // branch
        if parent_idx < len(nodes):
            tree.nodes[ids[idx]].parent = ids[parent_idx]
    tree.meta["elapsed"] = round(time.time() - t0, 2)
    return tree


def build_tree(engine, world: dict, npc: dict, depth: int, strategy: str = "levelwise",
               branch: int = 2, seed: int = 0, chunk_size: int = 1) -> Tree:
    if strategy == "oneshot":
        return build_oneshot(engine, world, npc, depth, branch, seed)
    if strategy == "nodewise":
        return build_levelwise(engine, world, npc, depth, branch, seed, chunk_size=1)
    return build_levelwise(engine, world, npc, depth, branch, seed, chunk_size=chunk_size)
