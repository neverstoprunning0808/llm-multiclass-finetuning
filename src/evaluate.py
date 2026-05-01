import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import AppConfig, load_config


def evaluate(config: AppConfig):

    new_reviews = [
        "The food was amazing!",
        "The food was good but the price was a bit too high.",
        "OK!",
        "The restaurant was dirty.",
        "Decent experience, but nothing special.",
    ]

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    new_tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.tokenizer_saved_path)

    inputs = new_tokenizer(
        new_reviews,
        padding=config.tokenizer.padding,
        truncation=config.tokenizer.truncation,
        max_length=config.tokenizer.max_length,
        return_tensors="pt",
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    new_model = AutoModelForSequenceClassification.from_pretrained(
        config.train.model_saved_path
    )
    new_model.to(device)

    new_model.eval()

    with torch.no_grad():
        outputs = new_model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

    star_ratings = predictions + 1

    for review, rating in zip(new_reviews, star_ratings):
        print(f"Review: {review}\nPredicted Star Rating: \
            {rating.item()}\n")


if __name__ == "__main__":
    config = load_config()
    evaluate(config)
