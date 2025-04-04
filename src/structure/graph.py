from typing import List

from src.structure.node import Node

class Graph:
    def __init__(self, nodes: List[Node]):
        self.nodes = nodes

    def __repr__(self):
        return f"Graph with size={len(self.nodes)}"

    def find_by_id(self, id: int):
        for node in self.nodes:
            if node.id == id:
                return node
        return None
