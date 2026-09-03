"""Environment-backed application settings without secret logging."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    database_url: str = "sqlite:///data/interim/brandpulse.db"
    model_path: Path = Path("artifacts/baseline/baseline_model.joblib")
    seed_data_path: Path | None = Path("data/interim/dataset_normalized.parquet")
    reports_dir: Path = Path("reports")
    cors_origins: tuple[str, ...] = ("http://localhost:3000",)

    @classmethod
    def from_env(cls) -> "Settings":
        seed_value = os.getenv("BRANDPULSE_SEED_DATA", "data/interim/dataset_normalized.parquet")
        origins = tuple(
            item.strip()
            for item in os.getenv("BRANDPULSE_CORS_ORIGINS", "http://localhost:3000").split(",")
            if item.strip()
        )
        return cls(
            database_url=os.getenv("DATABASE_URL", "sqlite:///data/interim/brandpulse.db"),
            model_path=Path(
                os.getenv("BRANDPULSE_MODEL_PATH", "artifacts/baseline/baseline_model.joblib")
            ),
            seed_data_path=Path(seed_value) if seed_value else None,
            reports_dir=Path(os.getenv("BRANDPULSE_REPORTS_DIR", "reports")),
            cors_origins=origins,
        )
