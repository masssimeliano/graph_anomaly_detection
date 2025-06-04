from dataclasses import dataclass

from src.helpers.config.datasets_config import *
from src.helpers.config.dir_config import *


@dataclass
class DataSet:
    def __init__(self, name: str, size: DataSetSize):
        self.name = name
        self.size = size
        self.location: Path = DATASETS_DIR / size.value / name

    def __repr__(self):
        return (f"DataSet(name={self.name}, "
                f"size={self.size.value}, "
                f"location={self.location})")

    def get_path(self) -> Path:
        return self.location

    def exists(self) -> bool:
        return self.location.exists()
