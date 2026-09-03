"""Pydantic request and response contracts."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    text: str = Field(max_length=5_000)
    company: str | None = Field(default=None, max_length=128)
    source_platform: str | None = Field(default=None, max_length=128)


class LabelPrediction(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    category: LabelPrediction
    sentiment: LabelPrediction
    priority: LabelPrediction
    needs_human_review: bool
    review_reasons: list[str]
    model_version: str


class BatchPredictionRequest(BaseModel):
    items: list[PredictionRequest] = Field(min_length=1, max_length=100)


class BatchPredictionResponse(BaseModel):
    items: list[PredictionResponse]


class CommentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    source_record_id: str | None
    text: str
    company: str | None
    source_platform: str | None
    source_url: str | None
    created_at_raw: str | None
    actual_category: str | None
    actual_sentiment: str | None
    actual_priority: str | None
    predicted_category: str | None
    category_confidence: float | None
    predicted_sentiment: str | None
    sentiment_confidence: float | None
    predicted_priority: str | None
    priority_confidence: float | None
    needs_human_review: bool
    review_reasons: list[str] = Field(default_factory=list)
    model_version: str | None


class CommentPage(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[CommentResponse]


class FeedbackRequest(BaseModel):
    comment_id: int | None = None
    text: str = Field(min_length=1, max_length=5_000)
    original_category: str | None = None
    corrected_category: str | None = None
    original_sentiment: str | None = None
    corrected_sentiment: str | None = None
    original_priority: str | None = None
    corrected_priority: str | None = None
    reviewer_note: str | None = Field(default=None, max_length=2_000)


class FeedbackResponse(BaseModel):
    id: int
    stored: bool
    automatic_retraining_triggered: bool = False
    created_at: datetime
