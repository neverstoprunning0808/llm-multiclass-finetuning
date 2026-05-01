import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    dataset_name: str = "yelp_review_full"
    train_size: int = 5000
    eval_size: int = 5000
    seed: int = 42
    output_path: str = "data/restaurant_ds"


class Tokenizer(BaseModel):
    data_path: str = "data/restaurant_ds"
    model_checkpoint: str = "distilbert-base-uncased"
    padding: str = "max_length"
    truncation: bool = True
    max_length: int = 512
    batch_size: int = 512
    output_path: str = "data/tokenized_ds"
    tokenizer_saved_path: str = "results/final_tokenizer"


class Train(BaseModel):
    model_checkpoint: str = "distilbert-base-uncased"
    num_labels: int = 5
    data_path: str = "data/tokenized_ds"
    output_dir: str = "results"
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 64
    per_device_eval_batch_size: int = 64
    num_train_epochs: int = 3
    weight_decay: float = 0.01
    logging_dir: str = "logs"
    logging_steps: int = 10
    save_steps: int = 500
    load_best_model_at_end: bool = True
    model_saved_path: str = "results/final_model"
    metrics_saved_path: str = "results/metrics"


class AppConfig(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    tokenizer: Tokenizer = Field(default_factory=Tokenizer)
    train: Train = Field(default_factory=Train)


def deep_merge(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = deep_merge(base[k], v)
        else:
            base[k] = v

    return base


def load_config(path: str = "params.yaml") -> AppConfig:
    default = AppConfig().model_dump()

    with open(path) as f:
        override = yaml.safe_load(f)

    return AppConfig(**deep_merge(default, override))
