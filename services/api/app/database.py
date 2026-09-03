"""SQLAlchemy engine, models, and local dataset seeding."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

import pandas as pd
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class Comment(Base):
    __tablename__ = "comments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_record_id: Mapped[str | None] = mapped_column(String(128), unique=True, nullable=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    company: Mapped[str | None] = mapped_column(String(128), index=True)
    source_platform: Mapped[str | None] = mapped_column(String(128), index=True)
    source_url: Mapped[str | None] = mapped_column(Text)
    created_at_raw: Mapped[str | None] = mapped_column(String(128))
    actual_category: Mapped[str | None] = mapped_column(String(128), index=True)
    actual_sentiment: Mapped[str | None] = mapped_column(String(64), index=True)
    actual_priority: Mapped[str | None] = mapped_column(String(64), index=True)
    predicted_category: Mapped[str | None] = mapped_column(String(128), index=True)
    category_confidence: Mapped[float | None] = mapped_column(Float)
    predicted_sentiment: Mapped[str | None] = mapped_column(String(64), index=True)
    sentiment_confidence: Mapped[float | None] = mapped_column(Float)
    predicted_priority: Mapped[str | None] = mapped_column(String(64), index=True)
    priority_confidence: Mapped[float | None] = mapped_column(Float)
    needs_human_review: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    review_reasons_json: Mapped[str] = mapped_column(Text, default="[]")
    model_version: Mapped[str | None] = mapped_column(String(128))
    inserted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    comment_id: Mapped[int | None] = mapped_column(ForeignKey("comments.id"), nullable=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    original_category: Mapped[str | None] = mapped_column(String(128))
    corrected_category: Mapped[str | None] = mapped_column(String(128))
    original_sentiment: Mapped[str | None] = mapped_column(String(64))
    corrected_sentiment: Mapped[str | None] = mapped_column(String(64))
    original_priority: Mapped[str | None] = mapped_column(String(64))
    corrected_priority: Mapped[str | None] = mapped_column(String(64))
    reviewer_note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


def build_session_factory(database_url: str) -> tuple[object, sessionmaker[Session]]:
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    engine = create_engine(database_url, pool_pre_ping=True, connect_args=connect_args)
    return engine, sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def session_dependency(factory: sessionmaker[Session]) -> Generator[Session, None, None]:
    with factory() as session:
        yield session


def seed_comments_if_empty(
    factory: sessionmaker[Session], seed_path: Path | None, inference_service: object
) -> int:
    if seed_path is None or not seed_path.is_file():
        return 0
    with factory() as session:
        if session.scalar(select(func.count()).select_from(Comment)):
            return 0
        frame = (
            pd.read_parquet(seed_path)
            if seed_path.suffix.lower() == ".parquet"
            else pd.read_csv(seed_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        )
        text_column = "text_normalized" if "text_normalized" in frame else "text_raw"
        texts = frame[text_column].fillna("").astype(str).tolist()
        predictions = inference_service.predict_many(texts)  # type: ignore[attr-defined]
        comments = []
        for (_, row), prediction in zip(frame.iterrows(), predictions):
            comments.append(
                Comment(
                    source_record_id=str(row["id"]),
                    text=str(row["text_raw"]),
                    company=str(row["company"]),
                    source_platform=str(row["source_platform"]),
                    source_url=str(row["source_url"]),
                    created_at_raw=str(row["created_at_raw"]),
                    actual_category=str(row["category"]),
                    actual_sentiment=str(row["sentiment"]),
                    actual_priority=str(row["priority"]),
                    predicted_category=prediction["category"]["label"],
                    category_confidence=prediction["category"]["confidence"],
                    predicted_sentiment=prediction["sentiment"]["label"],
                    sentiment_confidence=prediction["sentiment"]["confidence"],
                    predicted_priority=prediction["priority"]["label"],
                    priority_confidence=prediction["priority"]["confidence"],
                    needs_human_review=prediction["needs_human_review"],
                    review_reasons_json=__import__("json").dumps(prediction["review_reasons"]),
                    model_version=prediction["model_version"],
                )
            )
        session.add_all(comments)
        session.commit()
        return len(comments)
