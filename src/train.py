import json
import os

import torch
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

from config import AppConfig, load_config


def train(config: AppConfig):
    model = AutoModelForSequenceClassification.from_pretrained(
        config.train.model_checkpoint, num_labels=config.train.num_labels
    )

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    tokenized_dataset = load_from_disk(config.train.data_path)

    train_args = TrainingArguments(
        output_dir=config.train.output_dir,
        eval_strategy=config.train.eval_strategy,
        save_strategy=config.train.save_strategy,
        learning_rate=config.train.learning_rate,
        per_device_train_batch_size=config.train.per_device_train_batch_size,
        per_device_eval_batch_size=config.train.per_device_eval_batch_size,
        num_train_epochs=config.train.num_train_epochs,
        weight_decay=config.train.weight_decay,
        logging_dir=config.train.logging_dir,
        logging_steps=config.train.logging_steps,
        save_steps=config.train.save_steps,
        load_best_model_at_end=config.train.load_best_model_at_end,
        report_to="tensorboard",
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["val"],
    )

    trainer.train()

    model.save_pretrained(config.train.model_saved_path)

    metrics = trainer.evaluate()
    print(metrics)

    os.makedirs(config.train.metrics_saved_path, exist_ok=True)
    with open(f"{config.train.metrics_saved_path}/metrics.json", "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    config = load_config()
    train(config)
