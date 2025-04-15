from typing import List, Optional
from src.structure.node import Node

class Graph:
    def __init__(self, nodes: List[Node]):
        self.nodes = nodes
        self.node_map = {node.id: node for node in nodes}

    def __repr__(self):
        return f"Graph(size={len(self.nodes)})"

    def find_by_id(self, id: int) -> Optional[Node]:
        return self.node_map.get(id)

    def __len__(self):
        return len(self.nodes)

    def __iter__(self):
        return iter(self.nodes)

    def get_ids(self) -> List[int]:
        return list(self.node_map.keys())