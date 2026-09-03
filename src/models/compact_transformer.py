"""Offline fallback: a compact character-level multi-task Transformer encoder."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.evaluation.metrics import choose_review_threshold, classification_metrics
from src.models.multitask_transformer import TARGETS, build_label_maps, save_confusion_figure, set_seed

LOGGER = logging.getLogger("brandpulse.models.compact_transformer")


@dataclass(frozen=True)
class CompactConfig:
    max_length: int = 256
    min_character_frequency: int = 2
    embedding_size: int = 128
    attention_heads: int = 4
    feedforward_size: int = 256
    encoder_layers: int = 2
    dropout: float = 0.15
    batch_size: int = 32
    epochs: int = 25
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    patience: int = 5
    seed: int = 42


class CharacterVocabulary:
    PAD = "<PAD>"
    UNK = "<UNK>"
    CLS = "<CLS>"

    def __init__(self, tokens: dict[str, int]) -> None:
        self.tokens = tokens

    @classmethod
    def fit(cls, texts: list[str], minimum_frequency: int) -> "CharacterVocabulary":
        counts: Counter[str] = Counter(character for text in texts for character in text)
        tokens = {cls.PAD: 0, cls.UNK: 1, cls.CLS: 2}
        for character, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            if count >= minimum_frequency:
                tokens[character] = len(tokens)
        return cls(tokens)

    def encode(self, text: str, max_length: int) -> list[int]:
        values = [self.tokens[self.CLS]]
        values.extend(self.tokens.get(character, self.tokens[self.UNK]) for character in text)
        return values[:max_length]


class CharacterDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        frame: pd.DataFrame,
        vocabulary: CharacterVocabulary,
        label_maps: dict[str, dict[str, int]],
        max_length: int,
    ) -> None:
        self.sequences = [
            vocabulary.encode(text, max_length) for text in frame["text_normalized"].astype(str)
        ]
        self.labels = {
            target: [label_maps[target][str(value)] for value in frame[target]] for target in TARGETS
        }

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "input_ids": self.sequences[index],
            "labels": {target: labels[index] for target, labels in self.labels.items()},
        }


def collate_characters(rows: list[dict[str, Any]]) -> dict[str, Any]:
    maximum = max(len(row["input_ids"]) for row in rows)
    input_ids = torch.zeros((len(rows), maximum), dtype=torch.long)
    attention_mask = torch.zeros((len(rows), maximum), dtype=torch.bool)
    for index, row in enumerate(rows):
        length = len(row["input_ids"])
        input_ids[index, :length] = torch.tensor(row["input_ids"], dtype=torch.long)
        attention_mask[index, :length] = True
    labels = {
        target: torch.tensor([row["labels"][target] for row in rows], dtype=torch.long)
        for target in TARGETS
    }
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class CompactMultiTaskTransformer(nn.Module):
    def __init__(self, vocabulary_size: int, label_sizes: dict[str, int], config: CompactConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(vocabulary_size, config.embedding_size, padding_idx=0)
        self.position = nn.Embedding(config.max_length, config.embedding_size)
        layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_size,
            nhead=config.attention_heads,
            dim_feedforward=config.feedforward_size,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.encoder_layers)
        self.normalization = nn.LayerNorm(config.embedding_size)
        self.dropout = nn.Dropout(config.dropout)
        self.heads = nn.ModuleDict(
            {target: nn.Linear(config.embedding_size, size) for target, size in label_sizes.items()}
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        hidden = self.embedding(input_ids) + self.position(positions)
        hidden = self.encoder(hidden, src_key_padding_mask=~attention_mask)
        pooled = self.dropout(self.normalization(hidden[:, 0]))
        return {target: head(pooled) for target, head in self.heads.items()}


def class_weight_tensors(
    train: pd.DataFrame,
    label_maps: dict[str, dict[str, int]],
    device: torch.device,
    enabled: bool,
) -> dict[str, torch.Tensor]:
    output = {}
    for target, mapping in label_maps.items():
        counts = train[target].astype(str).value_counts()
        values = np.array([counts[label] for label in mapping], dtype=float)
        weights = 1.0 / np.sqrt(values) if enabled else np.ones_like(values)
        weights /= weights.mean()
        output[target] = torch.tensor(weights, dtype=torch.float32, device=device)
    return output


def evaluate(
    model: CompactMultiTaskTransformer,
    loader: DataLoader[Any],
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    probabilities = {target: [] for target in TARGETS}
    labels = {target: [] for target in TARGETS}
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            logits = model(**inputs)
            for target in TARGETS:
                probabilities[target].append(torch.softmax(logits[target], dim=-1).cpu().numpy())
                labels[target].append(batch["labels"][target].numpy())
    return (
        {target: np.concatenate(parts) for target, parts in probabilities.items()},
        {target: np.concatenate(parts) for target, parts in labels.items()},
    )


def validation_scores(
    probabilities: dict[str, np.ndarray], labels: dict[str, np.ndarray]
) -> dict[str, float]:
    return {
        target: float(
            f1_score(labels[target], probabilities[target].argmax(axis=1), average="macro", zero_division=0)
        )
        for target in TARGETS
    }


def train_one(
    train_loader: DataLoader[Any],
    validation_loader: DataLoader[Any],
    train: pd.DataFrame,
    vocabulary_size: int,
    label_maps: dict[str, dict[str, int]],
    config: CompactConfig,
    device: torch.device,
    seed: int,
    use_class_weights: bool,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], float, int]:
    set_seed(seed)
    model = CompactMultiTaskTransformer(
        vocabulary_size, {target: len(mapping) for target, mapping in label_maps.items()}, config
    ).to(device)
    weights = class_weight_tensors(train, label_maps, device, use_class_weights)
    losses = {target: nn.CrossEntropyLoss(weight=weights[target]) for target in TARGETS}
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    best_state: dict[str, torch.Tensor] | None = None
    best_score = -1.0
    best_epoch = 0
    stale_epochs = 0
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            labels = {target: values.to(device) for target, values in batch["labels"].items()}
            logits = model(**inputs)
            loss = sum(losses[target](logits[target], labels[target]) for target in TARGETS)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += float(loss.detach().cpu())
        scheduler.step()
        probabilities, observed = evaluate(model, validation_loader, device)
        scores = validation_scores(probabilities, observed)
        mean_score = float(np.mean(list(scores.values())))
        history.append(
            {
                "epoch": epoch,
                "training_loss": running_loss / len(train_loader),
                "validation_macro_f1": scores,
                "validation_mean_macro_f1": mean_score,
            }
        )
        if mean_score > best_score:
            best_score = mean_score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= config.patience:
            break
    assert best_state is not None
    return best_state, history, best_score, best_epoch


def _inverse_labels(label_maps: dict[str, dict[str, int]], target: str) -> np.ndarray:
    return np.array([label for label, _ in sorted(label_maps[target].items(), key=lambda item: item[1])])


def train_compact_transformer(
    processed_dir: Path,
    artifact_dir: Path,
    reports_dir: Path,
    config: CompactConfig,
) -> dict[str, Any]:
    frames = {
        name: pd.read_parquet(processed_dir / f"{name}.parquet")
        for name in ("train", "validation", "test")
    }
    train, validation, test = (frames[name] for name in ("train", "validation", "test"))
    label_maps = build_label_maps(train)
    vocabulary = CharacterVocabulary.fit(
        train["text_normalized"].astype(str).tolist(), config.min_character_frequency
    )
    datasets = {
        name: CharacterDataset(frame, vocabulary, label_maps, config.max_length)
        for name, frame in frames.items()
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    class_weight_trials = []
    cached_seed_state: dict[str, torch.Tensor] | None = None
    cached_seed_history: list[dict[str, Any]] | None = None
    cached_seed_best_epoch = 0
    selected_weighting = False
    best_trial_score = -1.0
    for use_weights in (False, True):
        generator = torch.Generator().manual_seed(config.seed)
        train_loader = DataLoader(
            datasets["train"],
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_characters,
            generator=generator,
            num_workers=0,
        )
        validation_loader = DataLoader(
            datasets["validation"], batch_size=config.batch_size * 2, shuffle=False, collate_fn=collate_characters
        )
        state, history, score, epoch = train_one(
            train_loader,
            validation_loader,
            train,
            len(vocabulary.tokens),
            label_maps,
            config,
            device,
            config.seed,
            use_weights,
        )
        class_weight_trials.append(
            {
                "use_inverse_sqrt_class_weights": use_weights,
                "best_validation_mean_macro_f1": score,
                "best_epoch": epoch,
            }
        )
        if score > best_trial_score:
            best_trial_score = score
            selected_weighting = use_weights
            cached_seed_state = state
            cached_seed_history = history
            cached_seed_best_epoch = epoch
    seed_runs = []
    production_state: dict[str, torch.Tensor] | None = None
    production_validation_score = -1.0
    production_seed = config.seed
    test_metrics_by_seed: dict[str, dict[str, Any]] = {}
    production_probabilities: dict[str, np.ndarray] | None = None
    production_labels: dict[str, np.ndarray] | None = None
    final_seeds = (config.seed, config.seed + 1, config.seed + 2)
    for seed in final_seeds:
        if seed == config.seed:
            assert cached_seed_state is not None and cached_seed_history is not None
            state, history, validation_score, best_epoch = (
                cached_seed_state,
                cached_seed_history,
                best_trial_score,
                cached_seed_best_epoch,
            )
        else:
            generator = torch.Generator().manual_seed(seed)
            train_loader = DataLoader(
                datasets["train"],
                batch_size=config.batch_size,
                shuffle=True,
                collate_fn=collate_characters,
                generator=generator,
                num_workers=0,
            )
            validation_loader = DataLoader(
                datasets["validation"], batch_size=config.batch_size * 2, shuffle=False, collate_fn=collate_characters
            )
            state, history, validation_score, best_epoch = train_one(
                train_loader,
                validation_loader,
                train,
                len(vocabulary.tokens),
                label_maps,
                config,
                device,
                seed,
                selected_weighting,
            )
        model = CompactMultiTaskTransformer(
            len(vocabulary.tokens),
            {target: len(mapping) for target, mapping in label_maps.items()},
            config,
        ).to(device)
        model.load_state_dict(state)
        validation_loader = DataLoader(
            datasets["validation"], batch_size=config.batch_size * 2, shuffle=False, collate_fn=collate_characters
        )
        test_loader = DataLoader(
            datasets["test"], batch_size=config.batch_size * 2, shuffle=False, collate_fn=collate_characters
        )
        validation_probabilities, validation_observed = evaluate(model, validation_loader, device)
        test_start = perf_counter()
        test_probabilities, test_observed = evaluate(model, test_loader, device)
        test_seconds = perf_counter() - test_start
        seed_target_metrics = {}
        for target in TARGETS:
            inverse = _inverse_labels(label_maps, target)
            truth = inverse[test_observed[target]]
            predictions = inverse[test_probabilities[target].argmax(axis=1)]
            seed_target_metrics[target] = classification_metrics(
                truth, predictions, test_probabilities[target], inverse, target
            )
        test_metrics_by_seed[str(seed)] = seed_target_metrics
        seed_runs.append(
            {
                "seed": seed,
                "best_epoch": best_epoch,
                "best_validation_mean_macro_f1": validation_score,
                "test_macro_f1": {
                    target: seed_target_metrics[target]["macro_f1"] for target in TARGETS
                },
                "test_inference_seconds": test_seconds,
                "history": history,
            }
        )
        if validation_score > production_validation_score:
            production_validation_score = validation_score
            production_seed = seed
            production_state = copy.deepcopy(state)
            production_probabilities = test_probabilities
            production_labels = test_observed
    assert production_state is not None and production_probabilities is not None and production_labels is not None
    production_model = CompactMultiTaskTransformer(
        len(vocabulary.tokens),
        {target: len(mapping) for target, mapping in label_maps.items()},
        config,
    ).to(device)
    production_model.load_state_dict(production_state)
    validation_loader = DataLoader(
        datasets["validation"], batch_size=config.batch_size * 2, shuffle=False, collate_fn=collate_characters
    )
    validation_probabilities, validation_observed = evaluate(
        production_model, validation_loader, device
    )
    targets = {}
    prediction_output = test[["id", "company", "source_platform", "text_raw", *TARGETS]].copy()
    for target in TARGETS:
        inverse = _inverse_labels(label_maps, target)
        validation_truth = inverse[validation_observed[target]]
        validation_predictions = inverse[validation_probabilities[target].argmax(axis=1)]
        threshold = choose_review_threshold(
            validation_truth, validation_predictions, validation_probabilities[target]
        )
        metrics = test_metrics_by_seed[str(production_seed)][target]
        targets[target] = {
            "validation_review_threshold": threshold,
            "test_metrics": metrics,
            "confusion_figure": save_confusion_figure(
                target, metrics, figures_dir, "compact_transformer"
            ),
            "three_seed_macro_f1_mean": float(
                np.mean([test_metrics_by_seed[str(seed)][target]["macro_f1"] for seed in final_seeds])
            ),
            "three_seed_macro_f1_std": float(
                np.std([test_metrics_by_seed[str(seed)][target]["macro_f1"] for seed in final_seeds])
            ),
        }
        predictions = inverse[production_probabilities[target].argmax(axis=1)]
        prediction_output[f"predicted_{target}"] = predictions
        prediction_output[f"confidence_{target}"] = production_probabilities[target].max(axis=1)
    checkpoint_path = artifact_dir / "model_state.pt"
    torch.save(production_state, checkpoint_path)
    vocabulary_path = artifact_dir / "vocabulary.json"
    vocabulary_path.write_text(json.dumps(vocabulary.tokens, ensure_ascii=False), encoding="utf-8")
    prediction_output.to_csv(
        reports_dir / "compact_transformer_test_predictions.csv", index=False, encoding="utf-8-sig"
    )
    metadata = {
        "model_version": "compact-char-transformer-v1",
        "model_type": "character_level_transformer_encoder_trained_from_scratch",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "label_maps": label_maps,
        "vocabulary_size": len(vocabulary.tokens),
        "vocabulary_fitted_on": "train_only",
        "target_loss_weights": {target: 1.0 for target in TARGETS},
        "class_weight_trials": class_weight_trials,
        "selected_class_weighting": selected_weighting,
        "class_weight_formula": "inverse_sqrt_train_frequency_normalized_to_mean_1"
        if selected_weighting
        else "none",
        "seed_runs": seed_runs,
        "production_seed_selected_on_validation_only": production_seed,
        "production_validation_mean_macro_f1": production_validation_score,
        "test_used_for_selection": False,
        "training_seconds": perf_counter() - start,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "model_state_size_bytes": checkpoint_path.stat().st_size,
        "pretrained_model_download_status": {
            "xlm-roberta-base": "blocked: weight CDN returned no body after successful metadata/tokenizer requests",
            "csebuetnlp/banglabert": "blocked: weight CDN returned no body after successful metadata/tokenizer requests",
        },
    }
    payload = {"metadata": metadata, "targets": targets}
    (artifact_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (reports_dir / "transformer_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def load_compact_artifact(
    artifact_dir: Path, device: torch.device | None = None
) -> tuple[CompactMultiTaskTransformer, CharacterVocabulary, dict[str, Any], torch.device]:
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    vocabulary = CharacterVocabulary(
        json.loads((artifact_dir / "vocabulary.json").read_text(encoding="utf-8"))
    )
    config = CompactConfig(**metadata["config"])
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CompactMultiTaskTransformer(
        len(vocabulary.tokens),
        {target: len(mapping) for target, mapping in metadata["label_maps"].items()},
        config,
    ).to(device)
    model.load_state_dict(
        torch.load(artifact_dir / "model_state.pt", map_location=device, weights_only=True)
    )
    model.eval()
    return model, vocabulary, metadata, device


def predict_texts(
    artifact_dir: Path, texts: list[str]
) -> list[dict[str, dict[str, str | float]]]:
    model, vocabulary, metadata, device = load_compact_artifact(artifact_dir)
    config = CompactConfig(**metadata["config"])
    rows = [{"input_ids": vocabulary.encode(text, config.max_length), "labels": {}} for text in texts]
    maximum = max(len(row["input_ids"]) for row in rows)
    input_ids = torch.zeros((len(rows), maximum), dtype=torch.long)
    mask = torch.zeros((len(rows), maximum), dtype=torch.bool)
    for index, row in enumerate(rows):
        length = len(row["input_ids"])
        input_ids[index, :length] = torch.tensor(row["input_ids"])
        mask[index, :length] = True
    with torch.inference_mode():
        logits = model(input_ids.to(device), mask.to(device))
    output = [dict() for _ in texts]
    for target in TARGETS:
        probabilities = torch.softmax(logits[target], dim=-1).cpu().numpy()
        inverse = _inverse_labels(metadata["label_maps"], target)
        indices = probabilities.argmax(axis=1)
        for row, index in enumerate(indices):
            output[row][target] = {
                "label": str(inverse[index]),
                "confidence": float(probabilities[row, index]),
            }
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/transformer/compact_v1"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    config = CompactConfig(epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)
    payload = train_compact_transformer(
        args.processed_dir.resolve(), args.artifact_dir.resolve(), args.reports_dir.resolve(), config
    )
    for target, values in payload["targets"].items():
        LOGGER.info(
            "%s production test macro-F1 %.4f (3-seed %.4f ± %.4f)",
            target,
            values["test_metrics"]["macro_f1"],
            values["three_seed_macro_f1_mean"],
            values["three_seed_macro_f1_std"],
        )


if __name__ == "__main__":
    main()
