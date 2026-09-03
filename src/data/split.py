"""Deterministic leakage-group construction and multi-target grouped splitting."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.audit import duplicate_signature

TARGET_COLUMNS = ("category", "sentiment", "priority")


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def build_leakage_groups(frame: pd.DataFrame) -> pd.Series:
    """Connect records sharing a source URL or normalized comment text."""

    if frame["id"].astype(str).duplicated().any():
        raise ValueError("IDs must be unique before leakage groups can be built")
    union_find = UnionFind(len(frame))
    for keys in (
        frame["source_url"].fillna("").astype(str).str.strip().str.casefold(),
        frame["text_raw"].fillna("").astype(str).map(duplicate_signature),
    ):
        first_seen: dict[str, int] = {}
        for position, key in enumerate(keys):
            if not key:
                continue
            if key in first_seen:
                union_find.union(position, first_seen[key])
            else:
                first_seen[key] = position
    members: dict[int, list[str]] = {}
    for position, record_id in enumerate(frame["id"].astype(str)):
        members.setdefault(union_find.find(position), []).append(record_id)
    names = {
        root: "grp_" + hashlib.sha256("|".join(sorted(ids)).encode("utf-8")).hexdigest()[:12]
        for root, ids in members.items()
    }
    return pd.Series(
        [names[union_find.find(position)] for position in range(len(frame))],
        index=frame.index,
        name="leakage_group",
    )


@dataclass(frozen=True)
class SplitConfig:
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 42
    search_attempts: int = 400

    def validate(self) -> None:
        total = self.train_fraction + self.validation_fraction + self.test_fraction
        if not np.isclose(total, 1.0):
            raise ValueError("Split fractions must sum to 1")
        if min(self.train_fraction, self.validation_fraction, self.test_fraction) <= 0:
            raise ValueError("Every split fraction must be positive")


def _group_matrix(frame: pd.DataFrame, group_column: str) -> tuple[pd.DataFrame, list[str]]:
    encoded_parts = []
    feature_names: list[str] = []
    for target in TARGET_COLUMNS:
        encoded = pd.get_dummies(frame[target].astype(str), prefix=target, dtype=int)
        encoded_parts.append(encoded)
        feature_names.extend(encoded.columns.tolist())
    encoded_frame = pd.concat(encoded_parts, axis=1)
    encoded_frame[group_column] = frame[group_column].values
    counts = encoded_frame.groupby(group_column, sort=True).sum()
    counts.insert(0, "_size", frame.groupby(group_column, sort=True).size())
    return counts, feature_names


def _solution_cost(
    sizes: np.ndarray,
    label_counts: np.ndarray,
    target_sizes: np.ndarray,
    target_labels: np.ndarray,
    global_labels: np.ndarray,
) -> float:
    size_cost = float(np.mean(((sizes - target_sizes) / np.maximum(target_sizes, 1.0)) ** 2))
    label_cost = float(
        np.mean(((label_counts - target_labels) / np.sqrt(np.maximum(target_labels, 1.0))) ** 2)
    )
    eligible = global_labels >= 3
    missing_penalty = float(((label_counts[:, eligible] == 0).sum()) * 0.30)
    return size_cost * 5.0 + label_cost + missing_penalty


def grouped_multitarget_split(
    frame: pd.DataFrame,
    config: SplitConfig | None = None,
    group_column: str = "leakage_group",
) -> pd.Series:
    """Assign whole leakage groups while approximating three label marginals."""

    config = config or SplitConfig()
    config.validate()
    group_counts, feature_names = _group_matrix(frame, group_column)
    groups = group_counts.index.to_numpy()
    sizes_by_group = group_counts["_size"].to_numpy(dtype=float)
    labels_by_group = group_counts[feature_names].to_numpy(dtype=float)
    fractions = np.array(
        [config.train_fraction, config.validation_fraction, config.test_fraction], dtype=float
    )
    target_sizes = fractions * len(frame)
    global_labels = labels_by_group.sum(axis=0)
    target_labels = fractions[:, None] * global_labels[None, :]
    rarity_weights = 1.0 / np.maximum(global_labels, 1.0)
    group_difficulty = sizes_by_group + (labels_by_group * rarity_weights).sum(axis=1) * len(frame)
    best_assignment: np.ndarray | None = None
    best_cost = float("inf")
    split_count = len(fractions)
    for attempt in range(config.search_attempts):
        rng = np.random.default_rng(config.seed + attempt)
        jitter = rng.uniform(0.0, 1e-3, size=len(groups))
        order = np.argsort(-(group_difficulty + jitter))
        sizes = np.zeros(split_count, dtype=float)
        label_counts = np.zeros((split_count, len(feature_names)), dtype=float)
        assignment = np.full(len(groups), -1, dtype=int)
        for group_index in order:
            candidates = []
            split_order = rng.permutation(split_count)
            for split_index in split_order:
                candidate_sizes = sizes.copy()
                candidate_labels = label_counts.copy()
                candidate_sizes[split_index] += sizes_by_group[group_index]
                candidate_labels[split_index] += labels_by_group[group_index]
                cost = _solution_cost(
                    candidate_sizes,
                    candidate_labels,
                    target_sizes,
                    target_labels,
                    global_labels,
                )
                candidates.append((cost, int(split_index)))
            _, chosen = min(candidates, key=lambda item: item[0])
            assignment[group_index] = chosen
            sizes[chosen] += sizes_by_group[group_index]
            label_counts[chosen] += labels_by_group[group_index]
        cost = _solution_cost(sizes, label_counts, target_sizes, target_labels, global_labels)
        if cost < best_cost:
            best_cost = cost
            best_assignment = assignment.copy()
    if best_assignment is None:
        raise RuntimeError("Unable to construct a grouped split")
    split_names = np.array(["train", "validation", "test"])
    mapping = {str(group): split_names[index] for group, index in zip(groups, best_assignment)}
    result = frame[group_column].astype(str).map(mapping)
    if result.isna().any():
        raise RuntimeError("Some records did not receive a split")
    return result.rename("split")


def leakage_check(frame: pd.DataFrame) -> dict[str, object]:
    """Return verifiable split-overlap and distribution diagnostics."""

    group_overlap = int(
        (frame.groupby("leakage_group")["split"].nunique(dropna=False) > 1).sum()
    )
    url_overlap = int(
        (frame.groupby(frame["source_url"].astype(str).str.strip().str.casefold())["split"].nunique() > 1).sum()
    )
    signature_overlap = int(
        (
            frame.assign(_signature=frame["text_raw"].astype(str).map(duplicate_signature))
            .groupby("_signature")["split"]
            .nunique()
            > 1
        ).sum()
    )
    return {
        "leakage_group_overlap_count": group_overlap,
        "source_url_overlap_count": url_overlap,
        "normalized_text_overlap_count": signature_overlap,
        "passed": group_overlap == 0 and url_overlap == 0 and signature_overlap == 0,
    }
