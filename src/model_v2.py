"""
model_v2.py
Improved model: fine-tuned DistilBERT (transfer learning) for sentiment classification.
This represents Model Version 2 -- improvements over the v1 LSTM baseline via:
  - Transfer learning from a pretrained transformer (distilbert-base-uncased)
  - Subword tokenization (handles OOV words far better than v1's whitespace vocab)
  - Dropout + AdamW + linear warmup scheduler for more stable fine-tuning
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "distilbert-base-uncased"


def get_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def get_model(num_labels: int = 2):
    return AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)


def tokenize_batch(tokenizer, texts, max_len=256):
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
