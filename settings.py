import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

data_dir = Path("data")
extension = ".txt"

# Disable tokenizers parallelism -- impacts torch dataloaders
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SplitType(str, Enum):
    train = "train"
    dev = "dev"
    test = "test"


@dataclass
class ModelConfig:
    # Model configs
    transformer_model: str = "distilbert-base-uncased"
    learning_rate: float = 1e-5
    batch_size: int = 32
    use_bias: bool = False
    model_dir: Path = Path("models")
    model_name: Optional[str] = None


@dataclass
class TrainerConfig:
    # Trainer config
    accelerator: str = "auto"
    fast_dev_run: bool = False
    max_epochs: int = 10


@dataclass
class Config:
    # Data configs
    split_paths: Dict[SplitType, Path] = field(
        default_factory=lambda: {
            SplitType.train: data_dir / f"{SplitType.train}{extension}",
            SplitType.dev: data_dir / f"{SplitType.dev}{extension}",
            SplitType.test: data_dir / f"{SplitType.test}{extension}",
        }
    )
    model: ModelConfig = ModelConfig()
    trainer: TrainerConfig = TrainerConfig()
    # Script configs
    fit: bool = True
    evaluate: bool = True
    num_workers: int = 4
    wandb_dir: Path = Path("wandb")
