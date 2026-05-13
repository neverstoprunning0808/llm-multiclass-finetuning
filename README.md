# 🍽️ Restaurant Review Sentiment Classification

Fine-tuned **DistilBERT** for 5-class sentiment classification on restaurant reviews, with a full **DVC pipeline** for reproducibility and a **Gradio** demo for inference.

---

## Overview

This project fine-tunes `distilbert-base-uncased` on the **Yelp Review Full** dataset to classify restaurant reviews into 5 sentiment categories (1–5 stars). The full workflow — from data preparation to tokenization to training — is managed by a **DVC pipeline**, ensuring reproducibility across environments.

| Component | Detail |
|-----------|--------|
| Base model | `distilbert-base-uncased` |
| Task | 5-class sentiment classification |
| Dataset | Yelp Review Full |
| Framework | HuggingFace Transformers + Trainer API |
| Pipeline | DVC |
| Demo | Gradio |

---

## Dataset

The project extracts subsets from the [Yelp Review Full](https://huggingface.co/datasets/Yelp/yelp_review_full) dataset from HuggingFace Datasets for training and testing. The data is filtered for reviews related to 'restaurant' and have short-medium length less than 500 words.

| Split | Size |
|-------|------|
| Train | 22,904 samples |
| Eval | 1,730 samples |
| Seed | 42 |
| # of class | 5 (star ratings 1–5) |

Raw data is saved to `data/restaurant_ds` after the data preparation stage.\
Tokenized data is save to `data/tokenized_ds` after the tokenizing stage.

---

## Model

- **Architecture:** DistilBERT (distilbert-base-uncased) + linear classification head
- **Max token length:** 512
- **Padding:** `max_length`
- **Truncation:** enabled
- **Labels:** 5 classes (0-4) corresponding to 1–5 star ratings

---

## Project Structure

```
.
├── data/
│   ├── restaurant_ds/       # Raw dataset (DVC-tracked)
│   └── tokenized_ds/        # Tokenized dataset (DVC-tracked)
├── results/
│   ├── final_model/         # Best model checkpoint
│   ├── final_tokenizer/     # Saved tokenizer
│   └── metrics/             # Evaluation metrics
├── src/
│   ├── prepare_data.py      # Stage 1: data loading & filtering
│   ├── tokenize_data.py     # Stage 2: tokenization
│   ├── train.py             # Stage 3: training
│   └── evaluate.py          # Stage 4: evaluation
├── params.yaml              # params config
├── config.yaml              # pydantic model's configuration
├── dvc.yaml                 # DVC pipeline definition
├── dvc.lock                 # DVC lock file
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone the repository

```bash
git clone git@github.com:neverstoprunning0808/llm-multiclass-finetuning.git
cd llm-multiclass-finetuning
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## DVC Pipeline

The pipeline has 4 sequential stages defined in `dvc.yaml`:

```
prepare_data  →  tokenize_data  →  train
```

### Run the full pipeline

```bash
dvc repro
```

### Run a specific stage

```bash
dvc repro prepare_data
dvc repro tokenize_data
dvc repro train
dvc repro evaluate
```

### Check pipeline status

```bash
dvc status
dvc dag        # Visualize the pipeline DAG
```

---

## Training

Training runs via `dvc repro`, but can also be triggered manually:

```bash
python src/train.py
```

### Training logs

Training logs are written to `logs/` and can be visualized with TensorBoard:

```bash
tensorboard --logdir logs
```

---

## Gradio Demo

A Gradio app is provided for interactive inference.

### Usage

Enter a restaurant review in the text box and the model will predict the sentiment rating (1–5 stars) with confidence scores.

---