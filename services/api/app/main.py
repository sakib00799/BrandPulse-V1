"""FastAPI entrypoint for BrandPulse-BD."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from functools import partial
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, or_, select, text
from sqlalchemy.orm import Session

from services.api.app.config import Settings
from services.api.app.database import Base, Comment, Feedback, build_session_factory, seed_comments_if_empty, session_dependency
from services.api.app.inference import InferenceService
from services.api.app.logging_config import configure_application_logging
from services.api.app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CommentPage,
    CommentResponse,
    FeedbackRequest,
    FeedbackResponse,
    PredictionRequest,
    PredictionResponse,
)

LOGGER = logging.getLogger("brandpulse.api")
configure_application_logging()


def _comment_response(comment: Comment) -> CommentResponse:
    payload = CommentResponse.model_validate(comment)
    try:
        payload.review_reasons = json.loads(comment.review_reasons_json)
    except json.JSONDecodeError:
        payload.review_reasons = ["invalid_stored_review_reason"]
    return payload


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    engine, session_factory = build_session_factory(settings.database_url)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        Base.metadata.create_all(engine)
        app.state.inference = InferenceService(settings.model_path)
        inserted = seed_comments_if_empty(session_factory, settings.seed_data_path, app.state.inference)
        LOGGER.info("API initialized; seeded_comments=%d model_version=%s", inserted, app.state.inference.model_version)
        yield
        engine.dispose()

    app = FastAPI(
        title="BrandPulse-BD API",
        version="0.1.0",
        description="Human-assisted Bangla/Banglish customer-feedback intelligence",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    figures_dir = settings.reports_dir / "figures"
    if figures_dir.is_dir():
        app.mount("/reports/figures", StaticFiles(directory=figures_dir), name="report-figures")
    get_session = partial(session_dependency, session_factory)

    @app.get("/health")
    def health(session: Session = Depends(get_session)) -> dict[str, Any]:
        session.execute(text("SELECT 1"))
        return {
            "status": "ok",
            "database": "ok",
            "model_loaded": hasattr(app.state, "inference"),
            "model_version": app.state.inference.model_version,
        }

    @app.get("/model-info")
    def model_info() -> dict[str, Any]:
        return app.state.inference.model_info()

    @app.post("/predict", response_model=PredictionResponse)
    def predict_one(request: PredictionRequest) -> dict[str, Any]:
        return app.state.inference.predict_one(request.text)

    @app.post("/batch-predict", response_model=BatchPredictionResponse)
    def predict_batch(request: BatchPredictionRequest) -> dict[str, Any]:
        return {"items": app.state.inference.predict_many([item.text for item in request.items])}

    @app.get("/comments", response_model=CommentPage)
    def comments(
        company: str | None = None,
        platform: str | None = None,
        category: str | None = None,
        sentiment: str | None = None,
        priority: str | None = None,
        needs_human_review: bool | None = None,
        min_confidence: float | None = Query(default=None, ge=0.0, le=1.0),
        search: str | None = Query(default=None, max_length=200),
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=25, ge=1, le=100),
        session: Session = Depends(get_session),
    ) -> CommentPage:
        filters = []
        if company:
            filters.append(Comment.company == company)
        if platform:
            filters.append(func.lower(Comment.source_platform) == platform.casefold())
        if category:
            filters.append(Comment.predicted_category == category)
        if sentiment:
            filters.append(Comment.predicted_sentiment == sentiment)
        if priority:
            filters.append(Comment.predicted_priority == priority)
        if needs_human_review is not None:
            filters.append(Comment.needs_human_review == needs_human_review)
        if min_confidence is not None:
            filters.append(
                or_(
                    Comment.category_confidence >= min_confidence,
                    Comment.sentiment_confidence >= min_confidence,
                    Comment.priority_confidence >= min_confidence,
                )
            )
        if search:
            filters.append(func.lower(Comment.text).contains(search.casefold()))
        count_statement = select(func.count()).select_from(Comment).where(*filters)
        total = int(session.scalar(count_statement) or 0)
        statement = (
            select(Comment)
            .where(*filters)
            .order_by(Comment.id)
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        items = [_comment_response(comment) for comment in session.scalars(statement)]
        return CommentPage(total=total, page=page, page_size=page_size, items=items)

    @app.get("/review-queue", response_model=CommentPage)
    def review_queue(
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=25, ge=1, le=100),
        session: Session = Depends(get_session),
    ) -> CommentPage:
        base = Comment.needs_human_review.is_(True)
        total = int(session.scalar(select(func.count()).select_from(Comment).where(base)) or 0)
        statement = (
            select(Comment)
            .where(base)
            .order_by(Comment.predicted_priority.desc(), Comment.id)
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        return CommentPage(
            total=total,
            page=page,
            page_size=page_size,
            items=[_comment_response(comment) for comment in session.scalars(statement)],
        )

    @app.get("/analytics/overview")
    def analytics_overview(session: Session = Depends(get_session)) -> dict[str, Any]:
        total = int(session.scalar(select(func.count()).select_from(Comment)) or 0)

        def distribution(column: Any) -> list[dict[str, Any]]:
            statement = select(column, func.count()).group_by(column).order_by(func.count().desc())
            return [
                {"label": label or "Unknown", "count": int(count)}
                for label, count in session.execute(statement)
            ]

        high_priority = int(
            session.scalar(
                select(func.count()).select_from(Comment).where(Comment.predicted_priority == "High")
            )
            or 0
        )
        return {
            "total_comments": total,
            "high_priority_count": high_priority,
            "sentiment_distribution": distribution(Comment.predicted_sentiment),
            "category_distribution": distribution(Comment.predicted_category),
            "priority_distribution": distribution(Comment.predicted_priority),
            "company_distribution": distribution(Comment.company),
            "platform_distribution": distribution(Comment.source_platform),
        }

    @app.get("/analytics/trends")
    def analytics_trends() -> dict[str, Any]:
        return {
            "available": False,
            "series": [],
            "data_quality_notice": "All supplied timestamps are relative and no trustworthy collection timestamp is available; time trends are disabled.",
        }

    @app.get("/analytics/performance")
    def analytics_performance() -> dict[str, Any]:
        model_evaluation_path = settings.reports_dir / "model_evaluation.json"
        baseline_path = settings.reports_dir / "baseline_metrics.json"
        if not model_evaluation_path.is_file() or not baseline_path.is_file():
            raise HTTPException(status_code=503, detail="Evaluation artifacts are unavailable")
        evaluation = json.loads(model_evaluation_path.read_text(encoding="utf-8"))
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        return {
            "selected_model": evaluation["selected_model"],
            "comparison": evaluation["comparison"],
            "subgroups": evaluation["subgroups"],
            "latency": evaluation["latency"],
            "model_metadata": baseline["metadata"],
            "targets": {
                target: {
                    "test_metrics": baseline["targets"][target]["test_metrics"],
                    "review_threshold": baseline["targets"][target]["review_threshold"],
                    "confusion_matrix_url": f"/reports/figures/baseline_{target}_confusion_matrix.png",
                }
                for target in ("category", "sentiment", "priority")
            },
        }

    @app.post("/feedback", response_model=FeedbackResponse, status_code=201)
    def feedback(request: FeedbackRequest, session: Session = Depends(get_session)) -> FeedbackResponse:
        if request.comment_id is not None and session.get(Comment, request.comment_id) is None:
            raise HTTPException(status_code=404, detail="Comment not found")
        record = Feedback(**request.model_dump())
        session.add(record)
        session.commit()
        session.refresh(record)
        return FeedbackResponse(
            id=record.id,
            stored=True,
            automatic_retraining_triggered=False,
            created_at=record.created_at,
        )

    return app


app = create_app()
