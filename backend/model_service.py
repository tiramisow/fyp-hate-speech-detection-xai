from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from lime_explainer import NO_TOXICITY_EXPLANATION, explain_prediction, select_label_to_explain

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = (
    PROJECT_ROOT / "models/bert_multilabel_grid_search_final/best_bert_full_checkpoint.pt"
)
TOKENIZER_PATH = PROJECT_ROOT / "models/bert_multilabel_grid_search_final/bert_tokenizer"


class BertPredictor:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.labels: list[str] = []
        self.thresholds: dict[str, float] = {}
        self.max_length = 256

    def load(self) -> None:
        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
        if not TOKENIZER_PATH.exists():
            raise FileNotFoundError(f"Tokenizer not found: {TOKENIZER_PATH}")

        checkpoint = torch.load(CHECKPOINT_PATH, map_location=self.device, weights_only=False)

        self.labels = list(checkpoint["labels"])
        self.thresholds = {label: float(checkpoint["best_thresholds"][label]) for label in self.labels}
        self.max_length = int(checkpoint["best_config"]["max_length"])
        model_name = checkpoint["model_name"]

        self.tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH))
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(self.labels),
            problem_type="multi_label_classification",
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> dict:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model is not loaded")

        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with torch.no_grad():
            logits = self.model(**encoded).logits[0]

        probabilities = {label: float(prob) for label, prob in zip(self.labels, torch.sigmoid(logits))}
        predictions = {
            label: int(probabilities[label] >= self.thresholds[label]) for label in self.labels
        }
        predicted_labels = [label for label in self.labels if predictions[label] == 1]

        result = {
            "text": text,
            "probabilities": probabilities,
            "predictions": predictions,
            "predicted_labels": predicted_labels,
            "thresholds": self.thresholds,
        }

        if not predicted_labels:
            result.update(
                {
                    "explanation": NO_TOXICITY_EXPLANATION,
                    "explained_label": None,
                    "important_words": [],
                    "weights": [],
                    "effects": [],
                }
            )
            return result

        explained_label = select_label_to_explain(predicted_labels, probabilities)

        def predict_proba_for_lime(texts: list[str]) -> np.ndarray:
            if not texts:
                return np.empty((0, 2))

            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}

            with torch.no_grad():
                logits = self.model(**encoded).logits

            label_probs = torch.sigmoid(logits)[:, self.labels.index(explained_label)].cpu().numpy()
            return np.column_stack([1.0 - label_probs, label_probs])

        result.update(
            explain_prediction(
                text=text,
                label=explained_label,
                predict_proba_fn=predict_proba_for_lime,
            )
        )
        return result
