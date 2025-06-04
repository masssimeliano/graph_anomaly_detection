from dataclasses import dataclass
from typing import List

from src.structure.node import Node


@dataclass
class Graph:
    nodes: List['Node']

    def __repr__(self):
        return f"Graph(size={len(self.nodes)})"
