"""
Context Sufficiency Scorer.

A lightweight DeBERTa-based classifier that predicts whether retrieved context
is sufficient for answering a given query. Outputs one of three labels:
  0 = Insufficient (should abstain)
  1 = Moderate (needs strong model)
  2 = Sufficient (weak model can handle it)
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

# Label semantics
LABEL_INSUFFICIENT = 0  # Abstain — context is too weak
LABEL_MODERATE = 1       # Route to strong model
LABEL_SUFFICIENT = 2     # Route to weak model


class SufficiencyDataset(Dataset):
    """Dataset for training the sufficiency scorer."""

    def __init__(self, queries: list[str], contexts: list[str],
                 labels: list[int], tokenizer, max_length: int = 512):
        self.queries = queries
        self.contexts = contexts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = self.queries[idx]
        context = self.contexts[idx]
        label = self.labels[idx]

        # Encode as: [CLS] query [SEP] context [SEP]
        encoding = self.tokenizer(
            query, context,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "token_type_ids": encoding.get("token_type_ids", torch.zeros_like(encoding["input_ids"])).squeeze(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class SufficiencyScorer:
    """
    Context sufficiency scorer using DeBERTa.
    
    Given a query and retrieved passages, predicts a sufficiency score
    indicating whether the context is sufficient for answering the query.
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        num_labels: int = 3,
        device: str = "cpu",
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.num_labels = num_labels
        self.device = device
        self.max_length = max_length
        self.model = None
        self.tokenizer = None

    def load_model(self, from_pretrained: Optional[str] = None):
        """Load model — either base model for training or fine-tuned for inference."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_path = from_pretrained or self.model_name
        logger.info(f"Loading sufficiency scorer from: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name if not from_pretrained else from_pretrained
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=self.num_labels,
            problem_type="single_label_classification",
        ).to(self.device)

        logger.info(f"Scorer loaded: {sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M parameters")

    def generate_training_labels(self, instances) -> list[int]:
        """
        Generate sufficiency labels from strong/weak model comparison.
        
        Label assignment logic:
          - Both correct → 2 (sufficient, weak model is fine)
          - Strong correct, weak wrong → 1 (moderate, needs strong model)
          - Both wrong → 0 (insufficient, should abstain)
          - Weak correct, strong wrong → 2 (sufficient, treat as sufficient)
        """
        labels = []
        stats = {0: 0, 1: 0, 2: 0}

        for inst in instances:
            if inst.strong_correct and inst.weak_correct:
                label = LABEL_SUFFICIENT
            elif inst.strong_correct and not inst.weak_correct:
                label = LABEL_MODERATE
            elif not inst.strong_correct and not inst.weak_correct:
                label = LABEL_INSUFFICIENT
            else:  # weak correct, strong wrong (rare)
                label = LABEL_SUFFICIENT

            labels.append(label)
            stats[label] += 1

        total = len(labels)
        logger.info(
            f"Label distribution: "
            f"insufficient={stats[0]} ({stats[0]/total*100:.1f}%), "
            f"moderate={stats[1]} ({stats[1]/total*100:.1f}%), "
            f"sufficient={stats[2]} ({stats[2]/total*100:.1f}%)"
        )
        return labels

    def train(
        self,
        train_queries: list[str],
        train_contexts: list[str],
        train_labels: list[int],
        val_queries: list[str],
        val_contexts: list[str],
        val_labels: list[int],
        num_epochs: int = 5,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        output_dir: str = "./models/scorer",
    ) -> dict:
        """Train the sufficiency scorer."""
        from transformers import get_linear_schedule_with_warmup

        if self.model is None:
            self.load_model()

        # Create datasets
        train_dataset = SufficiencyDataset(
            train_queries, train_contexts, train_labels,
            self.tokenizer, self.max_length
        )
        val_dataset = SufficiencyDataset(
            val_queries, val_contexts, val_labels,
            self.tokenizer, self.max_length
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * warmup_ratio),
            num_training_steps=total_steps,
        )

        # Handle class imbalance
        label_counts = np.bincount(train_labels, minlength=self.num_labels)
        class_weights = torch.tensor(
            len(train_labels) / (self.num_labels * label_counts),
            dtype=torch.float32
        ).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        best_val_acc = 0
        training_history = {"train_loss": [], "val_loss": [], "val_acc": []}

        for epoch in range(num_epochs):
            # Training
            self.model.train()
            total_loss = 0
            for batch in train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                labels = batch.pop("labels")

                outputs = self.model(**batch)
                loss = criterion(outputs.logits, labels)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)

            # Validation
            val_loss, val_acc, val_preds = self._evaluate(val_loader, criterion)

            training_history["train_loss"].append(avg_train_loss)
            training_history["val_loss"].append(val_loss)
            training_history["val_acc"].append(val_acc)

            logger.info(
                f"Epoch {epoch + 1}/{num_epochs}: "
                f"train_loss={avg_train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, "
                f"val_acc={val_acc:.4f}"
            )

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save(output_dir)
                logger.info(f"New best model saved (val_acc={val_acc:.4f})")

        return training_history

    def _evaluate(self, data_loader: DataLoader, criterion) -> tuple[float, float, list[int]]:
        """Evaluate the model on a data loader."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in data_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                labels = batch.pop("labels")

                outputs = self.model(**batch)
                loss = criterion(outputs.logits, labels)
                total_loss += loss.item()

                preds = outputs.logits.argmax(dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().tolist())

        avg_loss = total_loss / len(data_loader)
        accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
        return avg_loss, accuracy, all_preds

    def predict(self, query: str, context: str) -> tuple[int, float]:
        """
        Predict sufficiency for a single query-context pair.
        
        Returns:
            (label, confidence): Predicted label and softmax probability.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        self.model.eval()
        encoding = self.tokenizer(
            query, context,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**encoding)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
            label = probs.argmax().item()
            confidence = probs[label].item()

        return label, confidence

    def predict_batch(self, queries: list[str], contexts: list[str],
                       batch_size: int = 32) -> list[tuple[int, float]]:
        """Batch prediction for multiple query-context pairs."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        self.model.eval()
        results = []

        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i + batch_size]
            batch_contexts = contexts[i:i + batch_size]

            encodings = self.tokenizer(
                batch_queries, batch_contexts,
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**encodings)
                probs = torch.softmax(outputs.logits, dim=-1)
                labels = probs.argmax(dim=-1)
                confidences = probs.gather(1, labels.unsqueeze(1)).squeeze()

            for label, conf in zip(labels.cpu().tolist(), confidences.cpu().tolist()):
                results.append((label, conf))

        return results

    def get_sufficiency_score(self, query: str, context: str) -> float:
        """
        Get a continuous sufficiency score (0-1).
        
        Computed as a weighted combination of class probabilities:
        score = 0 * P(insufficient) + 0.5 * P(moderate) + 1.0 * P(sufficient)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        self.model.eval()
        encoding = self.tokenizer(
            query, context,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**encoding)
            probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()

        # Weighted score: 0*P(insuff) + 0.5*P(moderate) + 1.0*P(sufficient)
        score = 0.0 * probs[0] + 0.5 * probs[1] + 1.0 * probs[2]
        return float(score)

    def save(self, path: str):
        """Save model and tokenizer."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Scorer saved to {path}")

    def load(self, path: str):
        """Load fine-tuned model."""
        self.load_model(from_pretrained=path)
