"""
data_loader.py
Loads and prepares the IMDB Movie Reviews sentiment dataset (binary: positive/negative).
Dataset source: Hugging Face `datasets` library -> "imdb"
(50,000 reviews, 25k train / 25k test, perfectly balanced classes)
"""

from datasets import load_dataset
import pandas as pd
import re


def clean_text(text: str) -> str:
    """Basic text cleaning: lowercase, strip HTML tags, remove extra whitespace."""
    text = re.sub(r"<.*?>", " ", text)          # remove HTML tags (IMDB reviews have <br/>)
    text = re.sub(r"http\S+", " ", text)         # remove URLs
    text = re.sub(r"[^a-zA-Z0-9\s.,!?']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def load_imdb(sample_size: int = None):
    """
    Loads the IMDB dataset and returns train/test pandas DataFrames
    with columns: ['text', 'label'] (0 = negative, 1 = positive).

    Args:
        sample_size: if set, subsamples each split for fast local/Colab testing.
    """
    dataset = load_dataset("imdb")

    train_df = pd.DataFrame(dataset["train"])
    test_df = pd.DataFrame(dataset["test"])

    if sample_size:
        train_df = train_df.sample(n=min(sample_size, len(train_df)), random_state=42).reset_index(drop=True)
        test_df = test_df.sample(n=min(sample_size // 4, len(test_df)), random_state=42).reset_index(drop=True)

    train_df["text"] = train_df["text"].apply(clean_text)
    test_df["text"] = test_df["text"].apply(clean_text)

    return train_df, test_df


if __name__ == "__main__":
    train_df, test_df = load_imdb(sample_size=2000)
    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
    print(train_df.head())
    print("Label distribution (train):\n", train_df["label"].value_counts())
