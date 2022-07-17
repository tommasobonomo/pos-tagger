from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

data_dir = Path("data")
extension = ".txt"


class SplitType(str, Enum):
    train = "train"
    dev = "dev"
    test = "test"


@dataclass
class Config:
    transformer_model: str = "distilbert-base-uncased"
    split_paths: dict[SplitType, Path] = field(
        default_factory=lambda: {
            SplitType.train: data_dir / f"{SplitType.train}{extension}",
            SplitType.dev: data_dir / f"{SplitType.dev}{extension}",
            SplitType.test: data_dir / f"{SplitType.test}{extension}",
        }
    )
