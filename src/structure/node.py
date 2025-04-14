from typing import List
from dataclasses import dataclass, field

@dataclass
class Node:
    id: int
    label: int
    neighbours: List['Node'] = field(default_factory=list)
    features: List[float] = field(default_factory=list)
    is_str_anomaly: bool = False
    is_attr_anomaly: bool = False

    def __repr__(self):
        neighbour_ids = [n.id for n in self.neighbours]
        return (f"Node(id={self.id}, label={self.label}, "
                f"neighbours={neighbour_ids}, "
                f"str_anom={self.is_str_anomaly}, attr_anom={self.is_attr_anomaly})")