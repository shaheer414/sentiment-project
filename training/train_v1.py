"""
training/train_v1.py
Trains the baseline LSTM sentiment classifier (Model Version 1) and logs
parameters, metrics, and the trained model to MLflow.

Run from project root:
    python training/train_v1.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import mlflow
import mlflow.pytorch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.data_loader import load_imdb
from src.model_v1 import Vocabulary, LSTMSentimentClassifier

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG = {
    "max_vocab_size": 20000,
    "max_len": 200,
    "embed_dim": 128,
    "hidden_dim": 128,
    "num_layers": 2,
    "dropout": 0.3,
    "batch_size": 32,
    "lr": 1e-3,
    "epochs": 3,
    "sample_size": 4000,   # subsample for fast training on CPU / Colab free tier
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class IMDBDataset(Dataset):
    def __init__(self, df, vocab, max_len):
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids = self.vocab.encode(self.texts[idx], self.max_len)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary")
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def main():
    mlflow.set_experiment("imdb-sentiment-classification")

    with mlflow.start_run(run_name="model_v1_lstm_baseline"):
        mlflow.log_params(CONFIG)

        print("Loading data...")
        train_df, test_df = load_imdb(sample_size=CONFIG["sample_size"])

        vocab = Vocabulary(max_vocab_size=CONFIG["max_vocab_size"]).build(train_df["text"])
        mlflow.log_param("vocab_size", len(vocab))

        train_ds = IMDBDataset(train_df, vocab, CONFIG["max_len"])
        test_ds = IMDBDataset(test_df, vocab, CONFIG["max_len"])
        train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"])

        model = LSTMSentimentClassifier(
            vocab_size=len(vocab),
            embed_dim=CONFIG["embed_dim"],
            hidden_dim=CONFIG["hidden_dim"],
            num_layers=CONFIG["num_layers"],
            dropout=CONFIG["dropout"],
        ).to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
        criterion = nn.CrossEntropyLoss()

        print("Training Model V1 (LSTM baseline)...")
        for epoch in range(CONFIG["epochs"]):
            model.train()
            total_loss = 0
            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            metrics = evaluate(model, test_loader)
            print(f"Epoch {epoch+1}/{CONFIG['epochs']} - loss: {avg_loss:.4f} - "
                  f"val_acc: {metrics['accuracy']:.4f} - val_f1: {metrics['f1']:.4f}")

            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            mlflow.log_metric("val_accuracy", metrics["accuracy"], step=epoch)
            mlflow.log_metric("val_precision", metrics["precision"], step=epoch)
            mlflow.log_metric("val_recall", metrics["recall"], step=epoch)
            mlflow.log_metric("val_f1", metrics["f1"], step=epoch)

        final_metrics = evaluate(model, test_loader)
        mlflow.log_metrics({f"final_{k}": v for k, v in final_metrics.items()})

        # Log and register the model
        mlflow.pytorch.log_model(model, "model", registered_model_name="imdb_sentiment_v1_lstm")

        # Save vocab alongside for inference reproducibility
        os.makedirs("artifacts", exist_ok=True)
        import json
        with open("artifacts/vocab_v1.json", "w") as f:
            json.dump(vocab.word2idx, f)
        mlflow.log_artifact("artifacts/vocab_v1.json")

        print("Model V1 training complete. Final metrics:", final_metrics)


if __name__ == "__main__":
    main()
