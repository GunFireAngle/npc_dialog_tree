"""对话树数据结构、校验与渲染。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict

_PAREN = re.compile(r"[（(][^）)]{0,24}[）)]|\*[^*]{0,24}\*")
_PREFIX = re.compile(r"^(NPC|旁白|台词|情绪)[：:]\s*")


def clean_text(s: str, max_len: int | None = None) -> str:
    """去掉模型常见的括号动作、Markdown、角色前缀，统一空白。"""
    s = _PAREN.sub("", str(s))
    s = _PREFIX.sub("", s.strip())
    s = s.replace("**", "").strip(" 　\"'“”‘’")
    s = re.sub(r"[ \t]+", "", s)
    if max_len:
        s = s[:max_len]
    return s.strip()


@dataclass
class Node:
    id: str
    depth: int
    line: str
    emotion: str = "neutral"
    options: list[str] = field(default_factory=list)
    parent: str | None = None
    parent_option: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Tree:
    world_id: str
    npc_name: str
    depth: int
    branch: int
    strategy: str
    nodes: dict[str, Node] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    # ---------- 构建 ----------
    def add(self, node: Node) -> None:
        self.nodes[node.id] = node
        self.order.append(node.id)

    def children(self, node_id: str) -> list[str]:
        return [n.id for n in self.nodes.values() if n.parent == node_id]

    def root(self) -> Node:
        return self.nodes[self.order[0]]

    def path(self, node_id: str) -> list[Node]:
        chain: list[Node] = []
        cur = self.nodes[node_id]
        while cur is not None:
            chain.append(cur)
            cur = self.nodes[cur.parent] if cur.parent else None
        return list(reversed(chain))

    # ---------- 校验 ----------
    def expected_nodes(self) -> int:
        return sum(self.branch ** i for i in range(self.depth))

    def checks(self) -> dict:
        n = len(self.nodes)
        exp = self.expected_nodes()
        leaf = [x for x in self.nodes.values() if x.depth == self.depth]
        internal = [x for x in self.nodes.values() if x.depth < self.depth]
        bad_branch = sum(1 for x in internal if len(x.options) != self.branch)
        empty_line = sum(1 for x in self.nodes.values() if not x.line.strip())
        depth_max = max((x.depth for x in self.nodes.values()), default=0)
        return {
            "n_nodes": n,
            "expected_nodes": exp,
            "completeness": round(n / exp, 4) if exp else 0.0,
            "structure_valid": int(n == exp and bad_branch == 0 and empty_line == 0),
            "bad_branch_nodes": bad_branch,
            "empty_line_nodes": empty_line,
            "depth_achieved": depth_max,
            "depth_ok": int(depth_max == self.depth),
            "n_leaf": len(leaf),
        }

    def to_dict(self) -> dict:
        return {
            "world_id": self.world_id,
            "npc_name": self.npc_name,
            "depth": self.depth,
            "branch": self.branch,
            "strategy": self.strategy,
            "root_id": self.order[0] if self.order else None,
            "nodes": [self.nodes[i].to_dict() for i in self.order],
            "meta": self.meta,
        }

    @staticmethod
    def from_dict(d: dict) -> "Tree":
        t = Tree(
            world_id=d["world_id"],
            npc_name=d["npc_name"],
            depth=d["depth"],
            branch=d.get("branch", 2),
            strategy=d.get("strategy", "levelwise"),
            meta=d.get("meta", {}),
        )
        for nd in d["nodes"]:
            t.add(Node(**nd))
        return t

    # ---------- 渲染 ----------
    def render_text(self) -> str:
        """把树渲染成缩进的对话脚本，供人工阅读与 LLM 评审使用。"""
        lines: list[str] = []

        def walk(nid: str, indent: int) -> None:
            nd = self.nodes[nid]
            pad = "  " * indent
            lines.append(f"{pad}[L{nd.depth}] NPC（{nd.emotion}）：{nd.line}")
            if nd.depth < self.depth:
                for i, opt in enumerate(nd.options):
                    lines.append(f"{pad}  └ 选项{i + 1}：{opt}")
                    kids = [c for c in self.children(nid)]
                    if i < len(kids):
                        walk(kids[i], indent + 2)

        walk(self.order[0], 0)
        return "\n".join(lines)

    def render_plain(self) -> str:
        """按层铺开的紧凑文本，token 更省。"""
        by_level: dict[int, list[Node]] = {}
        for nid in self.order:
            nd = self.nodes[nid]
            by_level.setdefault(nd.depth, []).append(nd)
        out = []
        for d in sorted(by_level):
            out.append(f"== 第{d}层 ==")
            for nd in by_level[d]:
                opts = " / ".join(nd.options) if nd.options else "（结束）"
                out.append(f"{nd.id} {nd.line} [选项: {opts}]")
        return "\n".join(out)


def parse_nodes_json(text: str) -> list[dict]:
    """从模型输出中尽量稳健地抽取 nodes 数组。"""
    s = text.strip()
    if "```" in s:
        parts = s.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("{"):
                s = p
                break
    start = s.find("{")
    if start == -1:
        raise ValueError("no json object found")
    depth = 0
    end = -1
    for i, ch in enumerate(s[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end == -1:
        raise ValueError("unbalanced json")
    obj = json.loads(s[start:end])
    nodes = obj.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("missing nodes array")
    cleaned = []
    for nd in nodes:
        if not isinstance(nd, dict):
            continue
        cleaned.append(
            {
                "id": str(nd.get("id", "")),
                "line": str(nd.get("line", "")).strip(),
                "emotion": str(nd.get("emotion", "neutral")).strip() or "neutral",
                "options": [str(o).strip() for o in nd.get("options", []) if str(o).strip()],
            }
        )
    if not cleaned:
        raise ValueError("empty nodes")
    return cleaned
