"""
node.py
This file contains architecture class "Node", which contains information about node in graph.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Node:
    id: int
    label: int
    is_str_anomaly: bool
    is_attr_anomaly: bool
    neighbours: List["Node"] = field(default_factory=list)
    features: List[float] = field(default_factory=list)

    def __repr__(self):
        return (
            f"Node(id={self.id}, "
            f"label={self.label}, "
            f"neighbours={[node_neighbour.id for node_neighbour in self.neighbours]}, "
            f"str_anom={self.is_str_anomaly}, "
            f"attr_anom={self.is_attr_anomaly})"
        )
