from enum import Enum
from pathlib import Path
from src.helpers.config import DATASETS_DIR


class DataSetSize(Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"

class DataSet:
    def __init__(self, name: str, size: DataSetSize):
        self.name = name
        self.size = size
        self.location: Path = DATASETS_DIR / size.value / name

    def __repr__(self):
        return f"DataSet(name={self.name}, size={self.size.value}, location={self.location})"