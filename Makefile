.PHONY: audit preprocess train-baseline train-transformer evaluate test dev docker

audit:
	python -m src.data.audit --input-dir . --output-dir reports --processed-dir data/processed

preprocess:
	python -m src.data.prepare --input data/processed/dataset_version_1.parquet --interim-dir data/interim --processed-dir data/processed --reports-dir reports --seed 42

train-baseline:
	python -m src.models.baseline --processed-dir data/processed --artifact-dir artifacts/baseline --reports-dir reports --seed 42

train-transformer:
	python -m src.models.compact_transformer --processed-dir data/processed --artifact-dir artifacts/transformer/compact_v1 --reports-dir reports --epochs 25 --batch-size 32 --seed 42

evaluate:
	python -m src.evaluation.report --processed-dir data/processed --baseline-artifact artifacts/baseline/baseline_model.joblib --reports-dir reports

test:
	python -m pytest -q
	npm --prefix apps/web run typecheck

dev:
	python -m uvicorn services.api.app.main:app --reload --port 8000

docker:
	docker compose up --build
