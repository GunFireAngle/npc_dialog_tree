"""提示词模板（集中管理，便于 notes/prompts 留档与论文引用）。"""

from __future__ import annotations

import json

SYSTEM = (
    "你是一名资深中文游戏叙事设计师，擅长为角色扮演游戏撰写性格鲜明的 NPC 对话。"
    "你只写角色真正说出口的话，绝不写动作、神态、旁白或括号说明。"
    "你输出的一切内容必须是可直接被程序解析的 JSON，不写解释、不加 Markdown 代码块。"
)

FEWSHOT = (
    '【输出示例】\n'
    '节点 x1 的策略：试探\n'
    '输入：{"id":"x1","depth":2,"history":"（玩家刚走近）｜ NPC：这儿的规矩是先报名字。｜ 玩家：我叫阿禾。"}\n'
    '输出：{"nodes":[{"id":"x1","line":"阿禾。名字不坏。你要打听的事，值不值得我开口？",'
    '"emotion":"试探","options":["我知道你缺人手。","我只想知道那天晚上的事。"]}]}\n'
)

# 对话策略（dialogue act）库：系统在扩写前先决定"这句要做什么"，再由模型实现为台词。
TACTICS: dict[str, str] = {
    "戒备": "用短句设防，先确认对方来历，绝不先透露自己的事",
    "试探": "抛一句带钩子的话，试探对方是否可信，暗示自己知道些内情",
    "回避": "把话题挡回去，用一件具体物件或一个借口转移",
    "松动": "态度松动一点，透露一条与自己目标相关的小线索",
    "揭示": "说出不愿说的往事的一部分，但故意不说完",
    "交易": "明确提出一个可以立刻兑现的交换条件",
    "强硬": "被逼到墙角，用冷话或威胁顶回去",
    "收尾": "结束这段交谈，留下一个未解的悬念",
}


def plan_tactic(level: int, target_depth: int, player_intent: str | None, key: str) -> str:
    """确定性的策略规划：玩家意图 → NPC 应对策略；越接近叶层越倾向揭示/收尾。"""
    import zlib

    h = zlib.crc32(key.encode("utf-8"))
    if level == 1:
        return "戒备"
    if level >= target_depth - 1:
        return ["揭示", "收尾", "松动"][h % 3]
    mapping = {
        "追问": ["回避", "松动"],
        "共情": ["松动", "揭示"],
        "施压": ["强硬", "回避"],
        "威胁": ["强硬", "收尾"],
        "交易": ["交易", "试探"],
        "转移": ["收尾", "回避"],
    }
    if player_intent in mapping:
        return mapping[player_intent][h % 2]
    return ["试探", "回避", "松动"][h % 3]


def plan_beats(world: dict, npc: dict, target_depth: int, seed: int = 0) -> list[dict]:
    """A1：确定性的全局节拍表——每层一个内容目标 + 允许解锁的事实 + 禁止提前透露的内容。

    只做"内容级规划"（这层要推进什么），不覆盖 plan_tactic 的"话术策略"。
    秘密只在最后一层解锁（A4 状态机），中层的每条 fact 各分配一层。
    """
    facts = [f for f in npc.get("facts", []) if f]
    secret = npc.get("secret", "")
    beats: list[dict] = []
    for lv in range(1, target_depth + 1):
        if lv == 1:
            goal, unlock = "确认来意，用一句本世界的日常细节把玩家挡在门外", ""
        elif lv == target_depth:
            goal, unlock = "被逼到无法回避，说出秘密的一部分，并留下代价或悬念", secret
        else:
            unlock = facts[(lv - 2) % len(facts)] if facts else ""
            goal = f"让玩家确知：{unlock}" if unlock else "推进关系，让玩家觉得这一轮有所得"
        later: list[str] = []
        for l2 in range(lv + 1, target_depth + 1):
            later.append(secret if l2 == target_depth else (facts[(l2 - 2) % len(facts)] if facts else ""))
        forbid = "；".join(dict.fromkeys(x for x in later if x))
        beats.append({"level": lv, "goal": goal, "unlock": unlock, "forbid": forbid})
    return beats


def npc_card(world: dict, npc: dict) -> str:
    facts = "\n".join(f"- {f}" for f in npc["facts"])
    return (
        f"【世界观】{world['name']}：{world['setting']}\n"
        f"【你扮演】{npc['name']}，{npc['role']}。\n"
        f"【目标】{npc['goal']}\n"
        f"【不愿说出口的事】{npc['secret']}\n"
        f"【说话方式】{npc['style']}\n"
        f"【价值观】{npc['values']}\n"
        f"【禁忌】{npc['taboos']}\n"
        f"【不可违背的设定】\n{facts}"
    )


def _frontier_block(frontier: list[dict]) -> str:
    """把每个待扩写节点渲染成清晰的对话转写，避免多节点串味。"""
    blocks = []
    for f in frontier:
        hist = f["history"]
        po = f.get("parent_option")
        # 玩家最后一句单独拎出来，避免小模型把它当成"该写的内容"而整句复述
        if po and hist.endswith(f"玩家：{po}") and "｜ 玩家：" in hist:
            hist = hist[: hist.rfind("｜ 玩家：")]
        lines = [x.strip() for x in hist.split("｜") if x.strip()]
        trans = []
        for ln in lines:
            if ln.startswith("NPC："):
                trans.append("NPC：" + ln[4:])
            elif ln.startswith("玩家："):
                trans.append("玩家：" + ln[3:])
            else:
                trans.append(ln)
        body = "\n".join("    " + t for t in trans) if trans else "    （玩家刚走近，还没开口）"
        if po:
            body += (
                f"\n    玩家刚说：「{po}」"
                f"\n    → 现在轮到 NPC 说话：只写 NPC 的回应，绝对不要复述玩家这句话。"
            )
        tactic = f.get("tactic", "")
        hint = f"｜本句策略：{tactic}（{TACTICS.get(tactic, '')}）" if tactic else ""
        if f.get("goal"):
            hint += f"｜本层目标：{f['goal']}"
        if f.get("leaf_mode") == "reveal":
            hint += "｜收尾方式：兑现（说出秘密的一部分）"
        blocks.append(f"节点 {f['id']}（第 {f['depth']} 层）{hint}\n{body}")
    return "\n\n".join(blocks)


def build_expand_messages(
    world: dict,
    npc: dict,
    frontier: list[dict],
    target_depth: int,
    branch: int = 2,
    recent_lines: list[str] | None = None,
    blacklist: list[str] | None = None,
    force_reveal: str | None = None,
) -> list[dict]:
    """逐层扩写：一次为若干前沿节点生成台词与分支选项。

    blacklist：A2 去重重采样时禁止复用的句子；force_reveal：A4 收尾重写时要求兑现的秘密。
    A1 的层次目标通过 frontier 节点的 goal 字段注入，不额外塞整张节拍表。
    """
    is_leaf = all(f["depth"] >= target_depth for f in frontier)
    recent = "\n".join(f"- {x}" for x in (recent_lines or [])[-12:]) or "- 无"
    banned = "\n".join(f"- {x}" for x in (blacklist or [])[-12:])
    banned_block = (
        f"\n\n【以下句子已被判为重复，必须彻底换一种说法（不得改写、不得换词复用）】\n{banned}"
        if banned else ""
    )
    reveal_block = (
        f"\n【必须兑现的秘密】这一句必须把下面内容的一部分说出口"
        f"（可以换成自己的话，但不得回避、不得含糊）：{force_reveal}"
        if force_reveal else ""
    )

    if is_leaf:
        end = "本句必须把【不愿说出口的事】的一部分说出口，并留下代价或悬念。"
        rules = (
            f"这是对话的最后一层（第 {target_depth} 层），每个节点只写一句收尾台词，options 必须为空数组 []。\n"
            f"{end}不要说教，不要总结全篇。"
        )
    else:
        rules = (
            f"每个节点写一句 NPC 台词，并给出恰好 {branch} 个玩家可选回应（options）。\n"
            "两个选项的意图必须明显不同，例如：追问事实 / 共情安抚 / 直接施压 / 提出交易 / 威胁 / 转移话题。\n"
            "选项要针对刚才这句台词，不能是泛泛的“继续”“好吧”。"
        )

    user = (
        f"{npc_card(world, npc)}\n\n"
        f"{FEWSHOT}\n"
        f"【已出现过的台词（不要重复或改写它们）】\n{recent}{banned_block}\n\n"
        f"【本次任务】当前是第 {frontier[0]['depth']} 层。为下面每个节点续写 NPC 的下一句台词"
        f"（注意：玩家台词只是上下文，你只能写 NPC 的话）：\n"
        f"{_frontier_block(frontier)}\n\n"
        f"{reveal_block}"
        f"【硬性要求】\n"
        f"1. 台词 20-45 个汉字，口语化，符合上面的说话方式；每句必须落到一个具体细节"
        f"（物件、地点、时间、数字或往事），不要空泛表态；\n"
        f"2. 严格实现该节点标注的【本句策略】，但不要把策略名称写进台词；{rules}\n"
        f"3. 只写 NPC 说出口的话：禁止括号、禁止动作/神态/心理描写、禁止“（沉默）”这类提示；"
        f"台词里不得出现玩家刚说的那句话；\n"
        f"4. 台词必须接住玩家最后那句话，且不得与已出现台词相同或近似；"
        f"禁止“我需要更多信息”“我知道你的处境”这类通用空话；\n"
        f"5. 选项不超过 20 个汉字，必须是玩家会说的话，且两个选项不得互相近义；\n"
        f"6. NPC 有自己的目的：不要反客为主采访玩家，不要连续提问；"
        f"每句台词要么推进【目标】，要么回避【不愿说出口的事】；\n"
        f"7. 不得与本 NPC 的设定、禁忌、世界观冲突；不得出现“AI”“模型”“对话树”等元词；\n"
        f"8. 台词必须落到实处，把节点上【本层目标】里的具体信息说出来或绕住它；"
        f"更深层的秘密不要提前说破。\n\n"
        f'【输出格式】只输出一个 JSON 对象：{{"nodes":[{{"id":"节点id","line":"台词","emotion":"情绪词","options":["选项1","选项2"]}}]}}\n'
        f"id 必须与输入完全一致，节点顺序与输入一致。"
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def build_oneshot_messages(world: dict, npc: dict, depth: int, branch: int = 2) -> list[dict]:
    """一次性生成整棵树（用于格式崩塌压力测试与消融）。"""
    n_nodes = sum(branch ** i for i in range(depth))
    user = (
        f"{npc_card(world, npc)}\n\n"
        f"【任务】生成一棵完整的 NPC 对话树：根节点为第 1 层，共 {depth} 层，"
        f"每个非叶节点恰好 {branch} 个玩家选项，总节点数必须为 {n_nodes}。\n"
        f"节点 id 从 n1 开始按层序编号：第 1 层 n1；第 2 层 n2、n3；第 3 层 n4..n7，以此类推。\n"
        f"每条台词不超过 45 字，选项不超过 20 字，两个选项意图必须不同，"
        f"第 {depth} 层为叶节点，options 为空数组。\n"
        f"只输出 JSON：{{\"nodes\":[{{\"id\":\"n1\",\"depth\":1,\"line\":\"...\",\"emotion\":\"...\",\"options\":[\"...\",\"...\"]}}]}}"
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def build_repair_messages(bad_output: str, err: str) -> list[dict]:
    user = (
        "上一条输出无法被 JSON 解析，错误信息：" + err + "\n"
        "请把它改写成严格合法的 JSON，只输出形如 "
        '{"nodes":[{"id":"...","line":"...","emotion":"...","options":["...","..."]}]} 的对象，'
        "不要解释，不要代码块。\n原始输出：\n" + bad_output[:2000]
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


JUDGE_SYSTEM = (
    "你是一名严格且公正的游戏叙事评测员。你只依据给定的世界观、NPC 设定与对话树文本打分，"
    "不推测作者意图，不因为篇幅长就给高分。你必须输出 JSON。"
)


JUDGE_CHECKS: dict[str, tuple[str, str]] = {
    "no_fact_conflict": ("consistency", "是否有台词与本 NPC 的身份、物品、经历等设定事实直接矛盾"),
    "no_cross_world_leak": ("consistency", "是否出现其他世界观专属的专名（别的世界名或别的 NPC 名）"),
    "no_out_of_world_terms": ("consistency", "是否出现世界观之外的现代词或元词（AI、模型、对话树、程序、系统、数据）"),
    "no_verbatim_repeat": ("persona_stability", "是否有两句及以上台词完全逐字相同"),
    "register_match": ("persona_stability", "台词是否符合该 NPC 的说话方式（句长、称呼、用词习惯）与人设，而不是路人腔"),
    "no_assistant_tone": ("persona_stability", "是否出现通用助手口吻、说教或总结（如“我理解你的感受”“希望这对你有帮助”）"),
    "options_distinct_intent": ("choice_meaningfulness", "同一节点的两个选项是否代表真正不同的立场（追问/共情/施压/交易/威胁/转移）"),
    "options_not_restated": ("choice_meaningfulness", "同一节点的两个选项是否只是换词重述同一句话"),
    "options_intent_recognisable": ("choice_meaningfulness", "选项是否都是玩家会说的具体话，而不是“继续”“好吧”这类无法判断意图的空话"),
}


def build_judge_check_messages(world: dict, npc: dict, tree_text: str) -> list[dict]:
    """把三个量表拆成可验证的二元子项：小模型的绝对打分严重趋中，子项判定更可用。"""
    lines = [f"- {k}：{desc}" for k, (_, desc) in JUDGE_CHECKS.items()]
    user = (
        f"【世界观】{world['name']}：{world['setting']}\n"
        f"【NPC 设定】{npc_card(world, npc)}\n\n"
        f"【待评对话树】\n{tree_text}\n\n"
        "【判定任务】逐个判断下面每项是否成立。成立填 true，不成立填 false，不要含糊。\n"
        + "\n".join(lines)
        + "\n\n判定标准：只有确实找到反例才填 false；找不到问题就填 true。\n"
        "【输出】只输出一个 JSON 对象，键与上面完全一致，外加 reason（不超过 50 字）：\n"
        '{"no_fact_conflict":true,"no_cross_world_leak":true,"no_out_of_world_terms":true,'
        '"no_verbatim_repeat":true,"register_match":true,"no_assistant_tone":true,'
        '"options_distinct_intent":true,"options_not_restated":true,"options_intent_recognisable":true,'
        '"reason":"..."}'
    )
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]


def build_judge_messages(world: dict, npc: dict, tree_text: str) -> list[dict]:
    user = (
        f"【世界观】{world['name']}：{world['setting']}\n"
        f"【NPC 设定】{npc_card(world, npc)}\n\n"
        f"【待评对话树】\n{tree_text}\n\n"
        "【评分说明】每项 1-5 分（整数），5 分最好。\n"
        "1. consistency（一致性）：台词与世界观、NPC 设定（身份、物品、经历）是否矛盾；"
        "不同分支之间是否自相矛盾；设定细节是否被错误改写。\n"
        "2. persona_stability（角色性格稳定性）：从头到尾是否保持同一个人格与说话方式；"
        "深层节点是否退化成泛泛的通用助手口吻或重复同一句话。\n"
        "3. choice_meaningfulness（玩家选择有意义性）：同一节点的两个选项是否代表真正不同的立场或后果，"
        "而不是换词重述；选项是否推动信息或关系变化。\n\n"
        '【输出】只输出 JSON：{"consistency":4,"persona_stability":4,"choice_meaningfulness":3,'
        '"reason":"不超过 60 字的依据"}'
    )
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]
