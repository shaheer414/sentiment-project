"""
model_v1.py
Baseline model: a simple Bidirectional LSTM classifier trained from scratch
on tokenized IMDB review text using a learned embedding layer.

This is intentionally simple (no pretrained embeddings, no transfer learning)
so that Model V2 (transformer-based, transfer learning) shows a clear improvement.
"""

import torch
import torch.nn as nn
from collections import Counter


class Vocabulary:
    """Simple whitespace-tokenizer vocabulary builder."""

    def __init__(self, max_vocab_size: int = 20000):
        self.max_vocab_size = max_vocab_size
        self.word2idx = {"<pad>": 0, "<unk>": 1}
        self.idx2word = {0: "<pad>", 1: "<unk>"}

    def build(self, texts):
        counter = Counter()
        for t in texts:
            counter.update(t.split())
        most_common = counter.most_common(self.max_vocab_size - 2)
        for i, (word, _) in enumerate(most_common, start=2):
            self.word2idx[word] = i
            self.idx2word[i] = word
        return self

    def encode(self, text, max_len=200):
        tokens = text.split()[:max_len]
        ids = [self.word2idx.get(tok, 1) for tok in tokens]
        ids += [0] * (max_len - len(ids))
        return ids

    def __len__(self):
        return len(self.word2idx)


class LSTMSentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 2)  # 2 classes: negative / positive

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        # concat last layer's forward and backward hidden states
        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        out = self.dropout(hidden_cat)
        return self.fc(out)
