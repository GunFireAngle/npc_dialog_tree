"""构造结构对齐用的 SFT 语料（程序化"银标准"数据）。

设计说明（论文中必须如实披露）：
- 这些样本不是人写的，而是由模板 + 世界观词槽程序化合成，目的是让 1.5B 模型学会
  ① 严格输出 {"nodes":[...]} 结构、② 两个选项意图互不相同、③ 台词长度与口吻约束、
  ④ 理解"前沿节点 + 历史路径"的输入契约。
- 内容创意仍由基础模型先验与提示承担，因此本训练属于"结构/风格对齐"，不是知识注入。
- 输入提示与实验阶段 build_expand_messages 完全一致，保证训练-推理同分布。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment.src.worlds import WORLDS  # noqa: E402
from experiment.src import prompts  # noqa: E402

LINES = {
    "戒备": [
        "站住。在{place}，生面孔先说明来意。",
        "你身上没有{thing}的味道，不像本地人。",
        "问{w0}？先告诉我你是谁的人。",
        "{w1}的事，不是随便谁都能打听的。",
    ],
    "试探": [
        "你问{w0}，是替谁问的？",
        "你既然提到了{thing}，那就把话说完整。",
        "你手上那道印子，跟{w1}有关吧。",
        "先说说，你想从我这儿换什么。",
    ],
    "回避": [
        "这话我当没听见。",
        "换个话题。{w0}的事我不谈。",
        "知道太多的人，在{place}活不长。",
        "{thing}？我说不清楚，也不打算说清楚。",
    ],
    "松动": [
        "……你倒是第一个不急着要答案的人。",
        "行，看在你没拿{w0}压我的份上。",
        "你说话的样子，像当年那个人。",
        "我把话说一半，剩下的你自己想。",
    ],
    "揭示": [
        "三十年的账，不是一句对不起能了结的。",
        "那天的{w1}，我确实看见了，但我没回头。",
        "他们让你看到的{thing}，全是被挑过的。",
        "我这条命，是从别人手里捡回来的。",
    ],
    "交易": [
        "你出得起什么？空话在这儿不好使。",
        "我帮你，但你得先把{w0}的事办妥。",
        "一口价：{thing}换一句实话。",
        "别拿承诺来换，我要能立刻兑现的。",
    ],
    "强硬": [
        "拿{w0}来压我？在这条街上不好使。",
        "你要真想撕破脸，我奉陪。",
        "再问一遍，我就当你是在找麻烦。",
        "我这条命不值钱，你也未必值。",
    ],
    "收尾": [
        "走吧。等你想清楚了再来。",
        "话到这儿。剩下的，看你自己的选择。",
        "记住你今天听到的，别记错我的脸。",
        "门在那边。下次来，带点能兑现的东西。",
    ],
}

OPTIONS = {
    "追问": ["你说的{w0}，到底指什么？", "后来呢？那天发生了什么？", "你为什么会知道这些？"],
    "共情": ["这事换谁都扛不住。", "我不急着要答案，你慢慢说。", "你一个人守了这么多年？"],
    "施压": ["别绕了，把话说明白。", "我现在就要一个答案。", "你在替谁遮掩？"],
    "交易": ["我帮你把{w1}修好，你告诉我实情。", "开个条件，我出得起。", "我拿{w0}换你一句话。"],
    "威胁": ["你要是不说，我只能上报了。", "别逼我动手。", "你确定要跟我耗下去？"],
    "转移": ["算了，就当我没问。", "先不说这个了。", "我去别处打听。"],
}

INTENTS = list(OPTIONS.keys())
EMOTIONS = list(LINES.keys())


def fill(tpl: str, world: dict, npc: dict, rng: random.Random) -> str:
    tone = world["tone_words"]
    return tpl.format(
        place=world["name"],
        w0=rng.choice(tone),
        w1=rng.choice(tone),
        w2=rng.choice(tone),
        thing=rng.choice(tone),
    )


def synth_levels(world: dict, npc: dict, depth: int, rng: random.Random):
    """逐层合成整棵树的文本；每层先按策略规划器决定对话行为，再按类别取台词模板。

    这样训练语料与推理时的输入契约（含"本句策略"字段）完全一致。
    """
    from experiment.src.metrics import intent_of

    counter = 1
    frontier = [
        {
            "id": f"n{counter}",
            "depth": 1,
            "history": "（玩家刚走近，还没开口）",
            "parent": None,
            "tactic": prompts.plan_tactic(1, depth, None, "root"),
        }
    ]
    counter += 1
    out = []
    for level in range(1, depth + 1):
        nxt = []
        for f in frontier:
            cat = f["tactic"]
            line = fill(rng.choice(LINES[cat]), world, npc, rng)
            if level < depth:
                intents = rng.sample(INTENTS, 2)
                opts = [fill(rng.choice(OPTIONS[i]), world, npc, rng) for i in intents]
            else:
                opts = []
            out.append({**f, "line": line, "emotion": cat, "options": opts})
            for i, o in enumerate(opts):
                nxt.append(
                    {
                        "id": f"n{counter}",
                        "depth": level + 1,
                        "history": f"{f['history']} ｜ NPC：{line} ｜ 玩家：{o}",
                        "parent": f["id"],
                        "tactic": prompts.plan_tactic(level + 1, depth, intent_of(o), f"{f['id']}-{i}"),
                    }
                )
                counter += 1
        frontier = nxt
    return out


def build_examples(n: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    examples = []
    while len(examples) < n:
        world = rng.choice(WORLDS)
        npc = world["npc"]
        depth = rng.choice([2, 3, 4, 5, 6])
        levels = synth_levels(world, npc, depth, rng)
        # 随机切一层作为监督目标，批量大小 1~6，模拟真实推理时的前沿节点数
        lv = rng.choice(range(1, depth + 1))
        level_nodes = [x for x in levels if x["depth"] == lv]
        rng.shuffle(level_nodes)
        batch = level_nodes[: rng.randint(1, min(2, len(level_nodes)))]
        recent = [x["line"] for x in levels if x["depth"] < lv]
        frontier = [
            {"id": x["id"], "depth": x["depth"], "history": x["history"], "tactic": x.get("tactic", "")}
            for x in batch
        ]
        msgs = prompts.build_expand_messages(world, npc, frontier, depth, 2, recent)
        completion = json.dumps(
            {"nodes": [{"id": x["id"], "line": x["line"], "emotion": x["emotion"], "options": x["options"]} for x in batch]},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        examples.append({"messages": msgs, "completion": completion, "world": world["id"], "depth": depth, "level": lv})
    return examples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "training" / "data"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = build_examples(args.n, args.seed)
    random.Random(0).shuffle(data)
    split = int(len(data) * 0.95)
    for name, part in (("train.jsonl", data[:split]), ("val.jsonl", data[split:])):
        with open(out / name, "w", encoding="utf-8") as f:
            for row in part:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"train={split} val={len(data) - split} -> {out}")
    print("示例 completion:", data[0]["completion"][:200])


if __name__ == "__main__":
    main()
