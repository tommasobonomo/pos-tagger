import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

data_dir = Path("data")
extension = ".txt"

# Disable tokenizers parallelism -- impacts torch dataloaders
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SplitType(str, Enum):
    train = "train"
    dev = "dev"
    test = "test"


@dataclass
class Config:
    split_paths: dict[SplitType, Path] = field(
        default_factory=lambda: {
            SplitType.train: data_dir / f"{SplitType.train}{extension}",
            SplitType.dev: data_dir / f"{SplitType.dev}{extension}",
            SplitType.test: data_dir / f"{SplitType.test}{extension}",
        }
    )
    transformer_model: str = "distilbert-base-uncased"
    learning_rate: float = 1e-5
    batch_size: int = 32
    use_bias: bool = False
