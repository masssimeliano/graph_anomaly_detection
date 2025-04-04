from typing import List

class Node:
    def __init__(self, id: int, label: int, neighbours: List['Node'], features: List[float], is_str_anomaly: bool, is_attr_anomaly: bool):
        self.id = id
        self.label = label
        self.neighbours = neighbours
        self.features = features
        self.is_str_anomaly = is_str_anomaly
        self.is_attr_anomaly = is_attr_anomaly

    def __repr__(self):
        return (
            f"Node(id={self.id}, "
            f"neighbours={[neighbour.id for neighbour in self.neighbours]}, "
            f"str_anom={self.is_str_anomaly}, "
            f"attr_anom={self.is_attr_anomaly})"
        )