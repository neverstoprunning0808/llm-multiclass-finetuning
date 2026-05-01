from datasets import load_from_disk
from transformers import AutoTokenizer

from config import AppConfig, load_config


def tokenize(config: AppConfig):
    ds = load_from_disk(config.tokenizer.data_path)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.model_checkpoint)

    def tokenizer_function(samples):
        return tokenizer(
            samples["text"],
            padding=config.tokenizer.padding,
            truncation=config.tokenizer.truncation,
            max_length=config.tokenizer.max_length,
        )

    # tokenized data
    tokenized_data = ds.map(
        tokenizer_function, batched=True, batch_size=config.tokenizer.batch_size,
        load_from_cache_file=False
    )

    # save to data folder data/tokenized_ds
    tokenized_data.save_to_disk(config.tokenizer.output_path)

    # save tokenizer
    tokenizer.save_pretrained(config.tokenizer.tokenizer_saved_path)


if __name__ == "__main__":
    config = load_config()
    tokenize(config)
