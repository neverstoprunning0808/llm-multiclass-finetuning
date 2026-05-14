import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_from_disk
from config import AppConfig, load_config
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import numpy as np
import gradio as gr
import json
import matplotlib.pyplot as plt

def launch_gr(config: AppConfig) -> None:
    interface = gr.Interface(
        fn = lambda text: evaluate(config, text),
        title = "Prediction",
        inputs = "text",
        outputs = "text"
    )

    interface.launch(share=True)

def evaluate(config: AppConfig) -> None:

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenized_dataset = load_from_disk(config.train.data_path)

    eval_tokenized = tokenized_dataset['val']

    eval_tokenized.set_format(
    type="torch",
    )

    labels = eval_tokenized[:]['label']


    eval_tokenized = eval_tokenized.remove_columns(['text', 'token_type_ids', 'label'])

    # input_ids = torch.tensor(eval_tokenized['input_ids'][:]).to(device)
    # attention_mask = torch.tensor(eval_tokenized['attention_mask'][:]).to(device)
    
    eval_dl = DataLoader(eval_tokenized, batch_size=32)

    new_model = AutoModelForSequenceClassification.from_pretrained(
        config.train.model_saved_path
    )
    new_model.to(device)

    new_model.eval()
    predictions = []

    with torch.no_grad():
        for batch in eval_dl:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = new_model(input_ids, attention_mask)
            logits = outputs.logits
            predictions.append(torch.argmax(logits, dim=-1).detach())

    predictions = torch.tensor(torch.cat(predictions)).cpu()
    accuracy = (predictions == labels).float().mean()

    print(f"The accuracy is: {accuracy: .4f}")

    metrics = {
    "accuracy": accuracy.item()
    }

    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    cm = confusion_matrix(labels.cpu().numpy(), predictions.cpu().numpy())

    plt.figure(figsize=(8, 8))
    plt.imshow(cm)

    plt.colorbar()

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    plt.savefig("results/confusion_matrix.png")


if __name__ == "__main__":
    config = load_config()

    evaluate(config)
    # launch_gr(config)

