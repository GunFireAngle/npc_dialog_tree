"""自动指标：结构、重复、多样性、人格漂移、设定冲突启发式。

这些指标不依赖人工，也不依赖模型，可在 CPU 上瞬间算完；
它们作为 LLM 评审与人工量表的客观补充，而非替代。
"""

from __future__ import annotations

import re
from collections import Counter

from .schema import Tree

# 玩家选项意图词典：用于估计"同一节点两个选项是否真的不同"
INTENT_LEXICON: dict[str, list[str]] = {
    "追问": ["为什么", "怎么", "什么", "哪", "是不是", "到底", "谁", "多少", "哪来", "凭什么"],
    "共情": ["我懂", "辛苦", "别急", "理解", "抱歉", "你可以", "我陪", "放心", "不容易"],
    "施压": ["必须", "现在", "立刻", "交代", "别废话", "说清楚", "告诉我", "回答问题"],
    "交易": ["交换", "条件", "帮你", "换", "报酬", "交易", "我出", "可以给", "互惠"],
    "威胁": ["否则", "后果", "别怪", "你会", "等着", "上报", "举报", "拆了", "毁掉"],
    "转移": ["算了", "不提了", "换个话题", "先不说", "以后再", "走吧", "告辞"],
}

ASSISTANT_LEAK = ["作为一个AI", "作为人工智能", "我可以帮助", "让我们", "首先", "总之", "希望这对你有帮助", "以下", "注意："]
STOPWORDS = set("的了是在和与就都而及及其这那你我他她它不没有一个我们你们他们可以因为所以但是不过已经一直非常真的")

CONFLICT_CUES = ["不是", "没有", "从不", "并不", "早已丢", "完好无损", "从来没有", "根本没有"]


def _norm(s: str) -> str:
    return re.sub(r"[\s，。！？、；：“”‘’…—,.!?;:\"'()（）]", "", s)


def _bigrams(s: str) -> set[str]:
    s = _norm(s)
    return {s[i : i + 2] for i in range(len(s) - 1)}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def intent_of(text: str) -> str:
    """选项意图；未命中词典时记为「其他」，使分母覆盖全部选项对。"""
    best, best_hits = "其他", 0
    for intent, kws in INTENT_LEXICON.items():
        hits = sum(1 for k in kws if k in text)
        if hits > best_hits:
            best, best_hits = intent, hits
    return best


def conflict_flags(tree: Tree, npc: dict) -> int:
    """启发式：在事实关键词附近出现否定词，计为一次疑似设定冲突。"""
    keywords = []
    for f in npc["facts"]:
        for tok in re.split(r"[，。、；：,;:]", f):
            tok = tok.strip()
            if 3 <= len(tok) <= 12:
                keywords.append(tok[-4:])
    keywords = [k for k in set(keywords) if k]
    n = 0
    for node in tree.nodes.values():
        for k in keywords:
            for cue in CONFLICT_CUES:
                if f"{cue}{k}" in node.line or f"{k}{cue}" in node.line:
                    n += 1
                    break
    return n


def numeric_drift(tree: Tree, npc: dict) -> int:
    """事实中的数量（如“十七把钥匙”“半壶水”）在台词里被改成别的数量。"""
    pat = re.compile(r"([一二三四五六七八九十百千两0-9]+)([把根张条个枚段块秒年层壶本册])")
    fact_nums = set()
    for f in npc["facts"]:
        for m in pat.finditer(f):
            fact_nums.add((m.group(1), m.group(2)))
    if not fact_nums:
        return 0
    drift = 0
    for node in tree.nodes.values():
        for m in pat.finditer(node.line):
            if (m.group(1), m.group(2)) not in fact_nums:
                drift += 1
                break
    return drift


META_TERMS = ["AI", "人工智能", "模型", "对话树", "程序", "系统", "数据", "算法"]

# 三个量表的确定性执行器：每项 1-5 分 = 1 + 4 × 通过率。
# 与 prompts.JUDGE_CHECKS 一一对应，便于与评审模型、人工评分三方对照。
RUBRIC_CHECKS: dict[str, list[str]] = {
    "consistency": ["no_fact_conflict", "no_cross_world_leak", "no_out_of_world_terms"],
    "persona_stability": ["no_verbatim_repeat", "register_match", "no_assistant_tone"],
    "choice_meaningfulness": ["options_distinct_intent", "options_not_restated", "options_intent_recognisable"],
}


def rubric_check_pass(m: dict) -> dict[str, bool]:
    return {
        "no_fact_conflict": m["fact_conflict_rate"] == 0,
        "no_cross_world_leak": m["cross_world_leak_rate"] == 0,
        "no_out_of_world_terms": m["meta_term_rate"] == 0,
        "no_verbatim_repeat": m["dup_line_rate"] < 0.10,
        "register_match": m["persona_hit_rate"] >= 0.30,
        "no_assistant_tone": m["persona_leak_rate"] == 0,
        "options_distinct_intent": m["intent_distinct_rate"] >= 0.50,
        "options_not_restated": m["option_dup_rate"] < 0.05,
        "options_intent_recognisable": m["intent_coverage"] >= 0.80,
    }


def rubric_proxy(m: dict) -> dict[str, float]:
    flags = rubric_check_pass(m)
    out = {}
    for scale, names in RUBRIC_CHECKS.items():
        out[scale] = round(1 + 4 * sum(flags[n] for n in names) / len(names), 3)
    out["composite"] = round(sum(out.values()) / len(RUBRIC_CHECKS), 3)
    return out


def _cross_world_leak(lines: list[str], world_id: str) -> int:
    """统计出现了其他世界观专属专名（世界名或 NPC 名）的台词条数。"""
    from experiment.src.worlds import WORLDS

    foreign = [t for w in WORLDS if w["id"] != world_id for t in (w["name"], w["npc"]["name"])]
    return sum(1 for l in lines if any(t in l for t in foreign))


def compute_metrics(tree: Tree, world: dict, npc: dict) -> dict:
    checks = tree.checks()
    nodes = list(tree.nodes.values())
    lines = [n.line for n in nodes if n.line.strip()]
    internal = [n for n in nodes if n.depth < tree.depth]

    # 重复
    norm_lines = [_norm(l) for l in lines]
    dup_line_rate = 1 - len(set(norm_lines)) / max(1, len(norm_lines))
    all_tok = "".join(norm_lines)
    grams = [all_tok[i : i + 3] for i in range(len(all_tok) - 2)]
    gram_dup_rate = 1 - len(set(grams)) / max(1, len(grams))
    distinct2 = len({all_tok[i : i + 2] for i in range(len(all_tok) - 1)}) / max(1, len(all_tok) - 1)

    # 选项
    opts = [o for n in internal for o in n.options]
    opt_dup_rate = 1 - len({_norm(o) for o in opts}) / max(1, len(opts))
    jac, intent_diff, n_pairs, intent_known = [], [], 0, 0
    for n in internal:
        if len(n.options) >= 2:
            a, b = n.options[0], n.options[1]
            jac.append(1 - _jaccard(_bigrams(a), _bigrams(b)))
            ia, ib = intent_of(a), intent_of(b)
            n_pairs += 1
            intent_diff.append(1.0 if ia != ib else 0.0)
            intent_known += sum(1 for i in (ia, ib) if i != "其他")
    option_semantic_distance = sum(jac) / len(jac) if jac else 0.0
    intent_distinct_rate = sum(intent_diff) / n_pairs if n_pairs else 0.0
    intent_coverage = intent_known / max(1, 2 * n_pairs)

    # 人格
    tone = world.get("tone_words", [])
    persona_hit_rate = (
        sum(1 for l in lines if any(t in l for t in tone)) / len(lines) if lines else 0.0
    )
    leak = sum(1 for l in lines if any(a in l for a in ASSISTANT_LEAK))
    persona_leak_rate = leak / max(1, len(lines))

    lengths = [len(l) for l in lines]
    mean_len = sum(lengths) / len(lengths) if lengths else 0.0
    std_len = (sum((x - mean_len) ** 2 for x in lengths) / len(lengths)) ** 0.5 if lengths else 0.0

    # 按层看人格/新颖度漂移
    by_level: dict[int, list[str]] = {}
    for n in nodes:
        by_level.setdefault(n.depth, []).append(n.line)
    level_tone = [
        sum(1 for l in ls if any(t in l for t in tone)) / max(1, len(ls))
        for _, ls in sorted(by_level.items())
    ]
    tone_drift = max(level_tone) - min(level_tone) if level_tone else 0.0

    conflicts = conflict_flags(tree, npc)
    drift = numeric_drift(tree, npc)

    out = {
        **checks,
        "dup_line_rate": round(dup_line_rate, 4),
        "gram_dup_rate": round(gram_dup_rate, 4),
        "distinct2": round(distinct2, 4),
        "option_dup_rate": round(opt_dup_rate, 4),
        "option_semantic_distance": round(option_semantic_distance, 4),
        "intent_distinct_rate": round(intent_distinct_rate, 4),
        "intent_coverage": round(intent_coverage, 4),
        "persona_hit_rate": round(persona_hit_rate, 4),
        "persona_leak_rate": round(persona_leak_rate, 4),
        "meta_term_rate": round(
            sum(1 for l in lines if any(t in l for t in META_TERMS)) / max(1, len(lines)), 4
        ),
        "cross_world_leak_rate": round(_cross_world_leak(lines, world["id"]) / max(1, len(lines)), 4),
        "tone_drift": round(tone_drift, 4),
        "mean_line_len": round(mean_len, 2),
        "std_line_len": round(std_len, 2),
        "fact_conflicts": conflicts,
        "fact_conflict_rate": round(conflicts / max(1, len(nodes)), 4),
        "numeric_drift": drift,
    }
    out.update(rubric_proxy(out))
    return out
