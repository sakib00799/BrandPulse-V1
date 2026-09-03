"""Fine-tune one shared transformer encoder with three classification heads."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from src.evaluation.metrics import choose_review_threshold, classification_metrics

LOGGER = logging.getLogger("brandpulse.models.multitask_transformer")
TARGETS = ("category", "sentiment", "priority")


@dataclass(frozen=True)
class TrainingConfig:
    model_name: str = "xlm-roberta-base"
    max_length: int = 128
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    max_grad_norm: float = 1.0
    dropout: float = 0.10
    seed: int = 42
    use_class_weights: bool = True
    gradient_checkpointing: bool = True


class EncodedTextDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        frame: pd.DataFrame,
        tokenizer: Any,
        label_maps: dict[str, dict[str, int]],
        max_length: int,
    ) -> None:
        self.encodings = tokenizer(
            frame["text_normalized"].astype(str).tolist(),
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        self.labels = {
            target: [label_maps[target][str(value)] for value in frame[target]] for target in TARGETS
        }

    def __len__(self) -> int:
        return len(self.labels[TARGETS[0]])

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = {key: values[index] for key, values in self.encodings.items()}
        item["labels"] = {target: values[index] for target, values in self.labels.items()}
        return item


class BatchCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        labels = {target: torch.tensor([row["labels"][target] for row in rows]) for target in TARGETS}
        features = [{key: value for key, value in row.items() if key != "labels"} for row in rows]
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        batch["labels"] = labels
        return batch


class MultiTaskTransformer(nn.Module):
    def __init__(
        self,
        model_name_or_config: str | Any,
        label_sizes: dict[str, int],
        dropout: float = 0.10,
        from_config: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = (
            AutoModel.from_config(model_name_or_config)
            if from_config
            else AutoModel.from_pretrained(model_name_or_config)
        )
        hidden_size = int(self.encoder.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict(
            {target: nn.Linear(hidden_size, label_sizes[target]) for target in TARGETS}
        )

    def forward(self, **inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.encoder(**inputs)
        pooled = self.dropout(outputs.last_hidden_state[:, 0])
        return {target: head(pooled) for target, head in self.heads.items()}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_label_maps(train: pd.DataFrame) -> dict[str, dict[str, int]]:
    maps = {}
    for target in TARGETS:
        labels = sorted(train[target].astype(str).unique().tolist())
        if target == "priority" and set(labels) == {"Low", "Medium", "High"}:
            labels = ["Low", "Medium", "High"]
        maps[target] = {label: index for index, label in enumerate(labels)}
    return maps


def compute_class_weights(
    train: pd.DataFrame, label_maps: dict[str, dict[str, int]], device: torch.device
) -> dict[str, torch.Tensor]:
    weights = {}
    for target, mapping in label_maps.items():
        counts = train[target].astype(str).value_counts()
        values = np.array([counts.get(label, 0) for label in mapping], dtype=float)
        if (values == 0).any():
            raise ValueError(f"Training split has a missing {target} class")
        inverse_sqrt = 1.0 / np.sqrt(values)
        inverse_sqrt /= inverse_sqrt.mean()
        weights[target] = torch.tensor(inverse_sqrt, dtype=torch.float32, device=device)
    return weights


def evaluate_loader(
    model: MultiTaskTransformer,
    loader: DataLoader[Any],
    device: torch.device,
    label_maps: dict[str, dict[str, int]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    model.eval()
    probabilities = {target: [] for target in TARGETS}
    labels = {target: [] for target in TARGETS}
    with torch.inference_mode():
        for batch in loader:
            label_batch = batch.pop("labels")
            inputs = {key: value.to(device) for key, value in batch.items()}
            autocast_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if device.type == "cuda"
                else nullcontext()
            )
            with autocast_context:
                logits = model(**inputs)
            for target in TARGETS:
                probabilities[target].append(torch.softmax(logits[target], dim=-1).cpu().numpy())
                labels[target].append(label_batch[target].numpy())
    return (
        {target: np.concatenate(values) for target, values in probabilities.items()},
        {target: np.concatenate(values) for target, values in labels.items()},
    )


def macro_f1_by_target(
    probabilities: dict[str, np.ndarray], labels: dict[str, np.ndarray]
) -> dict[str, float]:
    return {
        target: float(
            f1_score(labels[target], probabilities[target].argmax(axis=1), average="macro", zero_division=0)
        )
        for target in TARGETS
    }


def save_confusion_figure(
    target: str, metrics: dict[str, Any], figures_dir: Path, prefix: str
) -> str:
    figure, axis = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(
        confusion_matrix=np.asarray(metrics["confusion_matrix"]),
        display_labels=metrics["labels"],
    ).plot(ax=axis, cmap="Purples", colorbar=False, xticks_rotation=35)
    axis.set_title(f"{prefix} {target} confusion matrix")
    figure.tight_layout()
    path = figures_dir / f"{prefix}_{target}_confusion_matrix.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path.name


def _load_splits(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = tuple(
        pd.read_parquet(processed_dir / f"{name}.parquet")
        for name in ("train", "validation", "test")
    )
    for name, frame in zip(("train", "validation", "test"), frames):
        if frame["split"].nunique() != 1 or frame["split"].iloc[0] != name:
            raise ValueError(f"Invalid split marker in {name}.parquet")
    return frames  # type: ignore[return-value]


def train_transformer(
    processed_dir: Path,
    artifact_dir: Path,
    reports_dir: Path,
    config: TrainingConfig,
) -> dict[str, Any]:
    set_seed(config.seed)
    train, validation, test = _load_splits(processed_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Training %s on %s", config.model_name, device)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    label_maps = build_label_maps(train)
    label_sizes = {target: len(mapping) for target, mapping in label_maps.items()}
    datasets = {
        name: EncodedTextDataset(frame, tokenizer, label_maps, config.max_length)
        for name, frame in (("train", train), ("validation", validation), ("test", test))
    }
    collator = BatchCollator(tokenizer)
    generator = torch.Generator().manual_seed(config.seed)
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collator,
            generator=generator,
            num_workers=0,
        ),
        "validation": DataLoader(
            datasets["validation"], batch_size=config.batch_size * 2, shuffle=False, collate_fn=collator
        ),
        "test": DataLoader(
            datasets["test"], batch_size=config.batch_size * 2, shuffle=False, collate_fn=collator
        ),
    }
    model = MultiTaskTransformer(config.model_name, label_sizes, config.dropout).to(device)
    if config.gradient_checkpointing and hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable()
        if hasattr(model.encoder.config, "use_cache"):
            model.encoder.config.use_cache = False
    class_weights = compute_class_weights(train, label_maps, device)
    if not config.use_class_weights:
        class_weights = {target: torch.ones_like(weight) for target, weight in class_weights.items()}
    losses = {target: nn.CrossEntropyLoss(weight=class_weights[target]) for target in TARGETS}
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    updates_per_epoch = int(np.ceil(len(loaders["train"]) / config.gradient_accumulation_steps))
    total_updates = updates_per_epoch * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_updates * config.warmup_ratio)),
        num_training_steps=total_updates,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_score = -1.0
    best_epoch = -1
    history = []
    checkpoint_path = artifact_dir / "model_state.pt"
    training_start = perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for step, batch in enumerate(loaders["train"], start=1):
            label_batch = {target: value.to(device) for target, value in batch.pop("labels").items()}
            inputs = {key: value.to(device) for key, value in batch.items()}
            autocast_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if device.type == "cuda"
                else nullcontext()
            )
            with autocast_context:
                logits = model(**inputs)
                total_loss = sum(losses[target](logits[target], label_batch[target]) for target in TARGETS)
                total_loss = total_loss / config.gradient_accumulation_steps
            scaler.scale(total_loss).backward()
            running_loss += float(total_loss.detach().cpu()) * config.gradient_accumulation_steps
            should_step = step % config.gradient_accumulation_steps == 0 or step == len(loaders["train"])
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        validation_probabilities, validation_labels = evaluate_loader(
            model, loaders["validation"], device, label_maps
        )
        validation_scores = macro_f1_by_target(validation_probabilities, validation_labels)
        joint_score = float(np.mean(list(validation_scores.values())))
        history.append(
            {
                "epoch": epoch,
                "training_loss": running_loss / len(loaders["train"]),
                "validation_macro_f1": validation_scores,
                "validation_mean_macro_f1": joint_score,
            }
        )
        LOGGER.info("Epoch %d validation mean macro-F1 %.4f: %s", epoch, joint_score, validation_scores)
        if joint_score > best_score:
            best_score = joint_score
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    validation_probabilities, validation_labels = evaluate_loader(
        model, loaders["validation"], device, label_maps
    )
    thresholds = {}
    for target in TARGETS:
        inverse = np.array([label for label, _ in sorted(label_maps[target].items(), key=lambda item: item[1])])
        predictions = inverse[validation_probabilities[target].argmax(axis=1)]
        truth = inverse[validation_labels[target]]
        thresholds[target] = choose_review_threshold(truth, predictions, validation_probabilities[target])
    test_start = perf_counter()
    test_probabilities, test_labels = evaluate_loader(model, loaders["test"], device, label_maps)
    test_elapsed = perf_counter() - test_start
    target_results = {}
    prediction_output = test[["id", "company", "source_platform", "text_raw", *TARGETS]].copy()
    for target in TARGETS:
        inverse = np.array([label for label, _ in sorted(label_maps[target].items(), key=lambda item: item[1])])
        predictions = inverse[test_probabilities[target].argmax(axis=1)]
        truth = inverse[test_labels[target]]
        metrics = classification_metrics(
            truth, predictions, test_probabilities[target], inverse, target
        )
        target_results[target] = {
            "validation_review_threshold": thresholds[target],
            "test_metrics": metrics,
            "confusion_figure": save_confusion_figure(
                target, metrics, figures_dir, "xlmr_multitask"
            ),
        }
        prediction_output[f"predicted_{target}"] = predictions
        prediction_output[f"confidence_{target}"] = test_probabilities[target].max(axis=1)
    prediction_output.to_csv(reports_dir / "transformer_test_predictions.csv", index=False, encoding="utf-8-sig")
    tokenizer.save_pretrained(artifact_dir / "tokenizer")
    model.encoder.config.save_pretrained(artifact_dir / "encoder_config")
    runtime = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None,
        "training_seconds": perf_counter() - training_start,
        "test_inference_seconds": test_elapsed,
        "test_rows_per_second": len(test) / test_elapsed,
        "mean_test_latency_ms_per_row": test_elapsed * 1000 / len(test),
    }
    metadata = {
        "model_version": "xlmr-multitask-v1",
        "base_model": config.model_name,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_config": asdict(config),
        "target_loss_weights": {target: 1.0 for target in TARGETS},
        "class_weight_formula": "inverse_sqrt_train_frequency_normalized_to_mean_1"
        if config.use_class_weights
        else "none",
        "label_maps": label_maps,
        "best_epoch": best_epoch,
        "best_validation_mean_macro_f1": best_score,
        "history": history,
        "runtime": runtime,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "test_used_for_selection": False,
    }
    state_size = checkpoint_path.stat().st_size
    metadata["model_state_size_bytes"] = state_size
    payload = {"metadata": metadata, "targets": target_results}
    (artifact_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (reports_dir / "transformer_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def load_transformer_artifact(
    artifact_dir: Path, device: torch.device | None = None
) -> tuple[MultiTaskTransformer, Any, dict[str, Any], torch.device]:
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    label_maps = metadata["label_maps"]
    encoder_config = AutoConfig.from_pretrained(artifact_dir / "encoder_config")
    training_config = metadata["training_config"]
    model = MultiTaskTransformer(
        encoder_config,
        {target: len(mapping) for target, mapping in label_maps.items()},
        float(training_config["dropout"]),
        from_config=True,
    )
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(
        torch.load(artifact_dir / "model_state.pt", map_location=device, weights_only=True)
    )
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(artifact_dir / "tokenizer", use_fast=True)
    return model, tokenizer, metadata, device


def predict_texts(
    artifact_dir: Path, texts: list[str], max_length: int = 128
) -> list[dict[str, dict[str, float | str]]]:
    model, tokenizer, metadata, device = load_transformer_artifact(artifact_dir)
    batch = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in batch.items()}
    with torch.inference_mode():
        logits = model(**inputs)
    output = [dict() for _ in texts]
    for target in TARGETS:
        probabilities = torch.softmax(logits[target], dim=-1).cpu().numpy()
        inverse = np.array(
            [label for label, _ in sorted(metadata["label_maps"][target].items(), key=lambda item: item[1])]
        )
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
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/transformer/xlmr_multitask_v1"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--model-name", default="xlm-roberta-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    config = TrainingConfig(
        model_name=args.model_name,
        max_length=args.max_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
        use_class_weights=not args.no_class_weights,
        gradient_checkpointing=not args.no_gradient_checkpointing,
    )
    payload = train_transformer(
        args.processed_dir.resolve(), args.artifact_dir.resolve(), args.reports_dir.resolve(), config
    )
    for target, values in payload["targets"].items():
        LOGGER.info("%s test macro-F1: %.4f", target, values["test_metrics"]["macro_f1"])


if __name__ == "__main__":
    main()
