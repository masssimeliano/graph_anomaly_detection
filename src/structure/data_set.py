from pathlib import Path
from src.helpers.config import DATASETS_DIR, DataSetSize


class DataSet:
    def __init__(self, name: str, size: DataSetSize):
        self.name = name
        self.size = size
        self.location: Path = DATASETS_DIR / size.value / name

    def __repr__(self):
        return f"DataSet(name={self.name}, size={self.size.value}, location={self.location})"

    def get_path(self) -> Path:
        return self.location

    def exists(self) -> bool:
        return self.location.exists()
