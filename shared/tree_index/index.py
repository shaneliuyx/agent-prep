"""TreeIndex — flat-dict index over a hierarchical tree.json.

Three flat dicts built in one DFS at construction:

    id_map:        node_id -> node ref         (O(1) point lookup)
    parent_map:    node_id -> parent_id|None   (O(depth) ancestor walk)
    children_map:  node_id -> [child_ids]      (O(1) child iteration)

Picked over heavier alternatives (B+ tree, nested set, closure table,
SQLite, Postgres ltree) because at 50-500 nodes RAM-resident the
asymptotic differences are sub-microsecond; structure choice is
dominated by code-clarity + dep cost. Same shape as LlamaIndex's
`IndexGraph` (`Dict[str, List[str]]` adjacency).

Supports auto-merge / dynamic-granularity retrieval: when the agentic
loop has fetched 3+ siblings under the same parent, `subtree_ids(parent)`
returns all leaves under it for synthesis.
"""
from __future__ import annotations


class TreeIndex:
    """Flat-dict index over a hierarchical tree.json.

    Build once at retriever construction (one DFS, microseconds).
    All queries collapse to dict ops + bounded recursion.
    """

    def __init__(self, tree: dict):
        self.tree = tree
        self.id_map: dict[str, dict] = {}
        self.parent_map: dict[str, str | None] = {}
        self.children_map: dict[str, list[str]] = {}
        self._build(tree, parent_id=None)

    def _build(self, node: dict, parent_id: str | None) -> None:
        nid = node.get("node_id")
        if not nid:
            return
        self.id_map[nid] = node
        self.parent_map[nid] = parent_id
        self.children_map[nid] = [
            c["node_id"] for c in node.get("nodes", []) if c.get("node_id")
        ]
        for child in node.get("nodes", []):
            self._build(child, parent_id=nid)

    # ---- O(1) point lookup ----------------------------------------------
    def get(self, node_id: str) -> dict | None:
        return self.id_map.get(node_id)

    def has(self, node_id: str) -> bool:
        return node_id in self.id_map

    def __contains__(self, node_id: str) -> bool:
        return node_id in self.id_map

    def __len__(self) -> int:
        return len(self.id_map)

    def all_ids(self) -> list[str]:
        return list(self.id_map.keys())

    # ---- O(k) subtree walk (k = subtree size, unavoidable) -------------
    def subtree_ids(self, parent_id: str) -> list[str]:
        """Return parent + all descendants. DFS order, deterministic."""
        if parent_id not in self.id_map:
            return []
        out: list[str] = []
        stack = [parent_id]
        while stack:
            cur = stack.pop()
            out.append(cur)
            stack.extend(self.children_map.get(cur, []))
        return out

    def descendants(self, parent_id: str) -> list[str]:
        """Return descendants only (excludes parent itself)."""
        sub = self.subtree_ids(parent_id)
        return sub[1:] if sub else []

    def leaves_under(self, parent_id: str) -> list[str]:
        """Return only leaf nodes (no children) under parent_id."""
        return [
            nid for nid in self.subtree_ids(parent_id)
            if not self.children_map.get(nid)
        ]

    # ---- O(depth) ancestor walk ----------------------------------------
    def ancestors(self, node_id: str) -> list[str]:
        """Return ancestor chain from immediate parent up to root."""
        out: list[str] = []
        cur = self.parent_map.get(node_id)
        while cur is not None:
            out.append(cur)
            cur = self.parent_map.get(cur)
        return out

    def parent_of(self, node_id: str) -> str | None:
        return self.parent_map.get(node_id)

    def root_id(self) -> str | None:
        for nid, parent in self.parent_map.items():
            if parent is None:
                return nid
        return None
