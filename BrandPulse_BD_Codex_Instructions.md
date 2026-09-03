# Codex Project Instructions: BrandPulse-BD

## Role and objective

Act as a senior Machine Learning Engineer, NLP Engineer, Data Scientist, and full-stack AI developer. Build a complete, portfolio-ready project named **BrandPulse-BD: Bangla/Banglish Customer Feedback Intelligence Platform** using the CSV datasets that I provide.

The datasets contain public customer comments about companies such as Daraz and Grameenphone. The final product must turn unstructured Bangla, Banglish, English, and code-mixed feedback into useful business intelligence by predicting:

1. `category`
2. `sentiment`
3. `priority`

The project must include data auditing, preprocessing, model development, rigorous evaluation, backend API development, a usable frontend dashboard, Dockerization, testing, documentation, and deployment preparation.

This is a one-person, 30-day portfolio project. Keep the mandatory scope achievable and treat additional features as optional stretch goals.

---

## Data I will provide

There may be multiple CSV files. Automatically inspect the workspace and identify them by schema and content rather than assuming exact filenames.

### Possible raw-data schema

```csv
id,text,company,source_platform,source_url,created_at
1,"এইগুলা সব প্রমোশন বিক্রি বেশির ধান্দা যতগুলা ইউটিউবার মিস্ট্রি বক্স অর্ডার করে সবাই এই ইয়ারবাডস+স্মার্ট ওয়াচ পায় ভালো জিনিস পায়। আর আমরা অর্ডার করলে পাই না",Daraz,youtube,https://www.youtube.com/shorts/SslGPlUClpc?feature=share,8 months ago
```

### Possible labeled-data schema

```csv
id,text,company,source_platform,source_url,created_at,category,sentiment,priority
1,"এইগুলা সব প্রমোশন বিক্রি বেশির ধান্দা যতগুলা ইউটিউবার মিস্ট্রি বক্স অর্ডার করে সবাই এই ইয়ারবাডস+স্মার্ট ওয়াচ পায় ভালো জিনিস পায়। আর আমরা অর্ডার করলে পাই না",Daraz,youtube,https://www.youtube.com/shorts/SslGPlUClpc?feature=share,8 months ago,Delivery/Order,Neutral,Medium
```

Some labeled files may contain cleaned text rather than the exact original text. Some files may repeat the CSV header as a data row, contain quoted columns, use different encodings, or contain duplicate records.

---

## Non-negotiable rules

1. **Never overwrite, rename, move, or delete my original dataset files.** Treat them as read-only source data.
2. Store generated data only under `data/interim/` and `data/processed/`.
3. Do not silently fix, discard, relabel, or merge records. Record every transformation and exclusion with a reason.
4. Do not invent dataset statistics, evaluation scores, dates, labels, or business impact. Calculate and report actual values.
5. Do not use test-set information during preprocessing decisions, feature selection, hyperparameter tuning, threshold selection, or augmentation.
6. Do not put synthetic or LLM-generated examples in the final test set.
7. Do not expose API keys, tokens, credentials, usernames, or private information in code, logs, notebooks, GitHub, screenshots, or documentation.
8. Do not publish the complete raw dataset unless its licensing and platform terms clearly allow redistribution. Provide a small anonymized sample and data-preparation instructions instead.
9. Use the original minimally normalized text as the primary transformer input. Do not automatically remove Bangla stop words, punctuation, emojis, or negation words; they may be important for sentiment and intent.
10. Make only evidence-based claims. Clearly label assumptions, limitations, and optional recommendations.
11. Keep the system human-in-the-loop. `High` priority or low-confidence predictions must be reviewable by a human.
12. Ask me a question only when a missing decision truly blocks safe progress. Otherwise, make a reasonable documented assumption and continue.

---

## Phase 0: Workspace and data discovery

Before writing model or application code:

1. Inspect the repository/workspace structure.
2. Locate CSV/TSV/Excel/JSON dataset files.
3. Identify which files are raw, cleaned, processed, or labeled based on their schemas.
4. Detect encoding, delimiter, quoting, malformed lines, repeated headers, missing columns, and inconsistent column names.
5. Print a concise inventory containing:
   - filename
   - row count
   - column names
   - probable dataset role
   - encoding
   - parsing warnings
6. Create `reports/data_inventory.md`.
7. Present a short implementation plan before starting large changes.

If a Git repository already exists, preserve unrelated user changes. Do not initialize a second repository inside it.

---

## Phase 1: Data audit and dataset construction

Create a reproducible audit script, not only a notebook. The main command should be similar to:

```bash
python -m src.data.audit --input-dir data/raw --output-dir reports
```

The audit must analyze:

### Schema and missing data

- Missing `id`, `text`, `company`, `source_platform`, `source_url`, or labels
- Empty, whitespace-only, or extremely short text
- Rows where the CSV header was accidentally imported as data
- Mixed data types and invalid URLs
- Exact and normalized duplicates
- Duplicate IDs
- Same ID with conflicting text or labels
- Same text with conflicting labels
- Same `source_url` appearing across multiple records or files

### Label audit

- Unique values and frequencies for `category`, `sentiment`, and `priority`
- Capitalization and spelling variants such as `negative`, `Negative`, and `NEGATIVE`
- Minority and extremely rare classes
- Company-specific label vocabularies
- Invalid or missing labels
- Possible label contradictions, for example clearly negative text labeled `Neutral`

Do not automatically replace questionable labels. Export them to:

```text
reports/label_review_queue.csv
```

Include the original value, suspected problem, and suggested review reason.

### Text audit

Measure and visualize:

- text length by characters and tokens
- Bangla-script, Latin-script, English, and code-mixed proportions
- emoji, URL, number, and punctuation frequency
- top unigrams and character n-grams by class
- possible spam or repeated promotional comments
- near duplicates using normalized hashing and, if practical, embedding similarity

### Date warning

Values such as `8 months ago` are relative, not absolute dates. They cannot be converted correctly unless a reliable scrape/collection timestamp is available.

- Preserve the original value as `created_at_raw`.
- If a trustworthy `collected_at` or scrape timestamp exists, calculate an approximate absolute date and flag it as derived.
- If no reference timestamp exists, set `created_at_parsed` to null instead of inventing a date.
- Do not create time-trend claims from unresolved relative dates.

### Raw and labeled data reconciliation

Treat labeled rows as the supervised-learning source. Use raw files for provenance, verification, and possible future annotation.

When joining raw and labeled files:

1. Prefer stable IDs only after verifying ID uniqueness within the correct source scope.
2. Cross-check `company`, `source_platform`, and `source_url`.
3. Detect one-to-many or many-to-many joins before merging.
4. Preserve both `text_raw` and `text_cleaned` when they differ.
5. Generate a reconciliation report containing matched, unmatched, duplicated, and conflicting rows.

### Required outputs

Create:

```text
reports/data_audit.md
reports/data_quality.json
reports/label_review_queue.csv
reports/figures/
data/processed/dataset_version_1.csv
data/processed/dataset_version_1.parquet
data_card.md
```

The data card must document sources, columns, collection method, transformations, label definitions, privacy considerations, limitations, and redistribution restrictions.

---

## Phase 2: Preprocessing

Build preprocessing as reusable Python modules with unit tests.

### Minimal normalization for transformer models

- Unicode normalization
- normalize repeated whitespace
- remove invisible/control characters
- optionally replace URLs, phone-like strings, email addresses, order IDs, and account numbers with typed placeholders
- retain Bangla/English negation
- retain useful punctuation and emojis
- preserve the original text beside the normalized text

### Classical-model representation

Build a separate representation for baseline models using:

- character n-grams, preferably 3–5 or a tuned range
- optional word n-grams
- TF-IDF

Do not apply an English-only tokenizer or stemmer to Bangla text without evidence that it improves validation performance.

### Split strategy and leakage protection

Use a reproducible split with a fixed random seed, but prevent content leakage:

1. Group exact and near-duplicate comments together.
2. Group comments from the same `source_url` or source post/video together when practical.
3. If reliable absolute dates exist, compare a chronological test split with a grouped stratified split.
4. If a comment has generated paraphrases, keep the original and every paraphrase in the training split only.
5. Keep a final real, human-labeled test set untouched until model selection is complete.

Save split manifests containing record IDs so experiments are repeatable.

---

## Phase 3: Modeling

This is a **multi-task text-classification project** with three outputs:

- `category`: multi-class classification
- `sentiment`: multi-class classification
- `priority`: ordered multi-class classification, normally Low < Medium < High

### Baseline models

Train strong, interpretable baselines first:

1. Character TF-IDF + Logistic Regression for each target
2. Optionally compare Linear SVM if probability calibration is added properly

Use pipelines so preprocessing is fitted only on training data.

### Transformer experiments

Run a small, justified comparison rather than many uncontrolled models:

1. `xlm-roberta-base` as the primary multilingual/code-mixed candidate
2. `csebuetnlp/banglabert` as the Bangla-specialized comparison

Preferred architecture:

- one shared, trainable transformer encoder
- three separate classification heads
- weighted joint loss:

```text
L_total = w_category * L_category
        + w_sentiment * L_sentiment
        + w_priority * L_priority
```

Document how loss weights are selected. Compare the multi-task model with separate single-task baselines if time permits.

### Metadata experiment

The main production model should be text-first. Run a controlled ablation:

1. text only
2. text + `company`
3. text + `company` + `source_platform`

Check whether metadata improves genuine generalization or merely creates shortcuts. Report performance separately by company and platform. If possible, perform a cross-company stress test, but do not claim that a model generalizes to a completely unseen company unless the test design supports that conclusion.

### Class imbalance

Compare only justified techniques:

- class-weighted loss
- focal loss
- weighted sampling
- limited, human-reviewed augmentation for minority classes

Select the method using validation macro-F1 and minority-class recall, not accuracy alone.

### Reproducibility

- Fix and record Python, NumPy, and PyTorch seeds.
- Run the final selected configuration with at least three seeds if compute permits.
- Report mean and standard deviation.
- Save environment/package versions.
- Track experiments using MLflow or a clearly structured local experiment table.

---

## Phase 4: Evaluation and error analysis

Report metrics separately for all three targets.

### Required metrics

For `category` and `sentiment`:

- macro-F1 as the primary metric
- weighted-F1
- per-class precision, recall, and F1
- confusion matrix
- accuracy as a secondary metric

For `priority`:

- macro-F1
- per-class recall, especially `High`
- confusion matrix
- weighted Cohen's kappa or an ordinal error measure
- severe-error rate, especially `High` predicted as `Low`

For confidence quality:

- reliability diagram
- expected calibration error or Brier score
- selective accuracy/F1 versus coverage

For system performance:

- median and P95 inference latency
- model size
- memory usage if practical
- batch throughput

### Subgroup evaluation

Measure performance by:

- company
- source platform
- Bangla script versus Latin/Banglish versus code-mixed text
- short versus long comments
- common versus minority categories

### Error analysis

Manually inspect a reproducible sample of false positives and false negatives. Create an error taxonomy such as:

- sarcasm
- multiple complaints in one comment
- implicit sentiment
- label ambiguity
- insufficient context
- spelling/noise
- code mixing
- company-specific terminology

Create `reports/model_evaluation.md` with actual tables and figures. Do not hide poor results. Explain whether the problem is data quality, annotation ambiguity, model capacity, or deployment threshold.

---

## Phase 5: Application requirements

Build a production-style but achievable application.

### Backend

Use:

- Python
- FastAPI
- Pydantic
- SQLAlchemy
- PostgreSQL for deployment
- SQLite only as a convenient local fallback

Required endpoints:

```text
GET  /health
GET  /model-info
POST /predict
POST /batch-predict
GET  /comments
GET  /analytics/overview
GET  /analytics/trends
POST /feedback
```

`POST /predict` should accept at least:

```json
{
  "text": "Payment korechi kintu internet active hoy nai",
  "company": "Grameenphone",
  "source_platform": "youtube"
}
```

It should return:

```json
{
  "category": {"label": "...", "confidence": 0.0},
  "sentiment": {"label": "...", "confidence": 0.0},
  "priority": {"label": "...", "confidence": 0.0},
  "needs_human_review": true,
  "model_version": "..."
}
```

Use actual model probabilities or calibrated confidence. Do not return fake confidence values.

### Human review rules

Flag a row for review when:

- prediction confidence is below a validation-selected threshold
- predicted priority is `High`
- the three heads produce an operationally inconsistent combination
- input is empty, unsupported, extremely short, or out of distribution

Store human corrections as feedback without automatically retraining the production model.

### Frontend

Use Next.js, TypeScript, Tailwind CSS, and Recharts. Keep the design clean and recruiter-friendly.

Required pages:

1. **Overview dashboard**
   - total comments
   - sentiment distribution
   - high-priority count
   - top complaint categories
   - company and platform comparison

2. **Comment explorer**
   - filters for company, platform, category, sentiment, priority, and confidence
   - searchable text
   - link to `source_url`
   - actual and predicted labels when ground truth exists

3. **Live prediction**
   - user enters a new Bangla/Banglish comment
   - application displays all three predictions and confidence values
   - low-confidence warning

4. **Model performance**
   - metrics by model and target
   - confusion matrices
   - subgroup results
   - model version and training date

5. **Review queue**
   - low-confidence and high-priority comments
   - approve or correct predictions
   - save reviewer feedback

Do not display a time-series chart if `created_at` could not be converted reliably. Show a clear data-quality notice instead.

---

## Optional stretch feature: evidence-grounded insight assistant

Implement this only after every mandatory feature works.

Allow a business user to ask questions such as:

```text
What are the most common high-priority complaints about Daraz delivery?
```

The assistant must retrieve relevant comments and aggregate statistics from the database. Any generated summary must:

- cite the supporting record IDs and source URLs
- distinguish counts from qualitative examples
- refuse unsupported conclusions
- avoid exposing sensitive data
- never claim causality from observational comments

Use a provider interface so the LLM can be replaced. The core classification application must still work without an LLM API key.

---

## Phase 6: Engineering quality

Create a maintainable repository similar to:

```text
brandpulse-bd/
├── apps/
│   └── web/
├── services/
│   └── api/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│   └── inference/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── reports/
│   └── figures/
├── tests/
├── configs/
├── notebooks/
├── model_card.md
├── data_card.md
├── docker-compose.yml
├── Makefile
├── README.md
└── .env.example
```

Requirements:

- modular Python code rather than notebook-only logic
- type hints for application code
- docstrings where they add value
- structured logging
- centralized configuration
- deterministic data-split manifests
- unit tests for preprocessing and validation
- API integration tests
- model-loading smoke test
- linting and formatting
- Dockerfiles for frontend and backend
- Docker Compose for local startup
- GitHub Actions for linting and tests
- no secrets or large model checkpoints committed to Git

Provide convenient commands, for example:

```bash
make audit
make preprocess
make train-baseline
make train-transformer
make evaluate
make test
make dev
```

Use the actual package manager and commands supported by the repository; do not add commands that do not work.

---

## Four-week implementation plan

### Week 1: Data and baseline

- discover and audit all files
- reconcile raw and labeled records
- finalize label mapping only after reporting conflicts
- build preprocessing modules
- create leakage-safe splits
- train TF-IDF baselines
- deliver data card and baseline evaluation

**Exit criterion:** reproducible processed dataset, split manifests, audit report, and baseline metrics.

### Week 2: Transformer and evaluation

- fine-tune XLM-R
- compare BanglaBERT if compute permits
- build multi-task heads
- perform class-imbalance experiments
- calibrate confidence and choose review thresholds
- conduct subgroup and error analysis

**Exit criterion:** selected model with reproducible evaluation and honest limitations.

### Week 3: API and dashboard

- build database schema
- implement FastAPI endpoints
- build Next.js dashboard and live prediction
- implement review queue and feedback capture
- add tests and local Docker setup

**Exit criterion:** complete application works locally from raw request to saved feedback.

### Week 4: Productionization and portfolio

- optimize latency if required
- add CI/CD and deployment configuration
- deploy or produce deployment-ready artifacts
- run end-to-end acceptance tests
- finish README, model card, diagrams, screenshots, and demo video plan
- tag a clean v1 release

**Exit criterion:** recruiter can open the repository, understand the results, run the application, and view a working demo.

---

## GitHub and portfolio requirements

The README must begin with:

1. a one-sentence business problem
2. a screenshot or short GIF
3. live demo and API documentation links, when available
4. an architecture diagram
5. a real results table

Include:

- problem and business relevance
- dataset summary and label definitions
- data-quality findings
- modeling experiments
- baseline versus transformer comparison
- per-company and per-platform evaluation
- error analysis
- application architecture
- local setup instructions that have been tested
- API examples
- privacy and licensing statement
- limitations and future work

Use placeholders only while developing. Before final delivery, replace them with actual results or clearly state that a result is unavailable.

Prepare two resume bullets using verified numbers only. A suitable pattern is:

```text
Built and deployed BrandPulse-BD, a Bangla/Banglish customer-feedback platform that predicts category, sentiment, and priority across [N] comments from [N] companies and [N] platforms using a multi-task transformer and FastAPI.

Improved category macro-F1 from [baseline] to [final result], implemented confidence-based human review, and deployed a Dockerized Next.js/FastAPI/PostgreSQL application with [measured] P95 inference latency.
```

Do not write unsupported claims such as increased revenue, reduced support time, or production adoption unless those effects were actually measured.

---

## Definition of done

The project is complete only when all mandatory items below are satisfied:

- original datasets remain unchanged
- data inventory and audit reports exist
- label conflicts and duplicate risks are documented
- processed dataset and reproducible splits exist
- at least one classical baseline is trained
- at least one transformer model is trained and evaluated
- category, sentiment, and priority metrics are reported separately
- leakage checks and subgroup evaluation are completed
- low-confidence/high-priority human review works
- FastAPI endpoints work and have tests
- frontend dashboard and live prediction work
- Docker Compose starts the local system successfully
- README, data card, and model card are complete
- no secrets or private raw data are exposed
- all claimed numbers come from generated evaluation artifacts

---

## How to work with me

At the beginning:

1. Show the detected dataset inventory.
2. Explain any schema ambiguity or serious data-quality risk.
3. Provide a concise, ordered plan.
4. Start with the audit instead of jumping directly to model training.

During implementation:

- give short progress updates
- state assumptions explicitly
- verify commands, tests, and generated files
- preserve unrelated existing work
- commit changes in logical units if Git is configured and I ask for commits

At the end of each phase, report:

- what was completed
- files created or changed
- tests and commands run
- actual results
- known limitations
- next recommended step

Begin by inspecting the available data files and repository. Do not train a model until the data audit, label audit, and leakage-safe split design are complete.
