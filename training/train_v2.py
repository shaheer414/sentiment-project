"""
training/train_v2.py
Trains the improved sentiment classifier (Model Version 2): fine-tuned DistilBERT
using transfer learning. Logs parameters, metrics, and the model to MLflow.

Improvements over Model V1 (LSTM baseline):
  - Transfer learning from pretrained distilbert-base-uncased
  - Subword tokenization (handles rare/unseen words much better)
  - Linear LR warmup schedule
  - Weight decay regularization via AdamW

Run from project root:
    python training/train_v2.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
import mlflow
import mlflow.pytorch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.data_loader import load_imdb
from src.model_v2 import get_tokenizer, get_model

CONFIG = {
    "model_name": "distilbert-base-uncased",
    "max_len": 256,
    "batch_size": 16,
    "lr": 2e-5,
    "weight_decay": 0.01,
    "epochs": 2,
    "warmup_ratio": 0.1,
    "sample_size": 4000,  # subsample for fast fine-tuning on Colab GPU
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class IMDBTorchDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.encodings = tokenizer(
            df["text"].tolist(),
            padding="max_length",
            truncation=True,
            max_length=max_len,
        )
        self.labels = df["label"].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary")
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def main():
    mlflow.set_experiment("imdb-sentiment-classification")

    with mlflow.start_run(run_name="model_v2_distilbert_finetuned"):
        mlflow.log_params(CONFIG)

        print("Loading data...")
        train_df, test_df = load_imdb(sample_size=CONFIG["sample_size"])

        tokenizer = get_tokenizer()
        model = get_model(num_labels=2).to(DEVICE)

        train_ds = IMDBTorchDataset(train_df, tokenizer, CONFIG["max_len"])
        test_ds = IMDBTorchDataset(test_df, tokenizer, CONFIG["max_len"])
        train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"])

        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
        total_steps = len(train_loader) * CONFIG["epochs"]
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * CONFIG["warmup_ratio"]),
            num_training_steps=total_steps,
        )

        print("Fine-tuning Model V2 (DistilBERT)...")
        for epoch in range(CONFIG["epochs"]):
            model.train()
            total_loss = 0
            for batch in train_loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                optimizer.zero_grad()
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
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

        # Save model + tokenizer for serving
        save_dir = "artifacts/model_v2"
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

        mlflow.pytorch.log_model(model, "model", registered_model_name="imdb_sentiment_v2_distilbert")
        mlflow.log_artifacts(save_dir, artifact_path="hf_model")

        print("Model V2 training complete. Final metrics:", final_metrics)


if __name__ == "__main__":
    main()
