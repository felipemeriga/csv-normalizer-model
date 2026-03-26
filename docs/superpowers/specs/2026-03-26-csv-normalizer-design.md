# CSV Bank Transaction Normalizer — Design Spec

## Goal

Build a system that normalizes CSV exports from multiple Brazilian banks into a standard schema. Code handles the deterministic pipeline (reading, validation, saving). A fine-tuned local model handles the intelligent part (merchant normalization, category classification, date parsing).

## Architecture

A Python pipeline where code handles all deterministic work (CSV reading, iteration, validation, output) and a fine-tuned TinyLlama 1.1B model handles the intelligent parts. One transaction at a time goes to the model — no chance of skipping rows.

## Output Schema

Each transaction normalized to:

| Field       | Type   | Example           |
|-------------|--------|-------------------|
| date        | string | "2024-03-15"      |
| merchant    | string | "Riachuelo"       |
| description | string | "RCHLO*Riachuelo" |
| amount      | float  | -159.90           |
| category    | string | "Shopping"        |

Categories (fixed enum): Transport, Food, Shopping, Transfer, Bills, Entertainment, Health, Education, Subscriptions, Other.

## Shared Prompt Template

The prompt template is the contract between training data formatting and inference. Defined once, used everywhere:

```
Normalize this bank transaction: {raw_row_text}
```

Where `raw_row_text` is the key-value pairs from the CSV row as they appear (e.g., `Data: 15/03/2024, Descrição: UBER *TRIP SP, Valor: -25.90`). The model responds with a single JSON object matching the output schema.

## Error Handling Strategy

When the model returns invalid output (malformed JSON, wrong category, missing fields):

1. **Retry once** with the same prompt
2. If still invalid, **log the failure** (row index, raw input, model output) and **skip the row**
3. Skipped rows are collected in a separate `errors.jsonl` file for manual review
4. Pipeline never halts on a single bad row — it processes all rows and reports success/failure counts at the end

## Amount Sign Convention

Negative = debit (money spent), positive = credit (money received). Code (not the model) handles:

- Stripping currency symbols (R$)
- Converting Brazilian number format (1.000,00 → 1000.00)
- Mapping separate debit/credit columns to a single signed amount

The model receives the already-cleaned numeric amount and passes it through.

## CSV Detection

`normalize.py` uses `csv.Sniffer` for delimiter detection and `chardet` for encoding detection. If detection fails, falls back to UTF-8 + comma. Logs a warning when falling back.

## Labeling Process

Labeling is manual. The user provides real bank CSVs in `data/raw/`. They manually create labeled examples in `data/labeled/labeled.jsonl` — each line is one transaction with all 5 output fields. Target: ~50-100 labeled examples as seed data for the synthetic generator.

## Components

### 1. Schema (`schema.py`)

Pydantic models defining the normalized output. Category is a fixed enum of 10 values. Validates model output before accepting it.

### 2. Synthetic Data Generator (`generate_synthetic.py`)

Takes labeled JSONL as seed data. Outputs synthetic examples as JSONL (`data/synthetic/synthetic.jsonl`) where each line contains `raw_text` (simulated CSV row) and `expected` (normalized output). Generates ~1,000-2,000 training examples by:

- Randomizing dates, amounts, merchant names
- Varying column names, delimiters (comma, semicolon, tab), date formats
- Varying header presence, column ordering, encoding (UTF-8, Latin-1)
- Adding noise: extra whitespace, missing fields, special characters

Purely rule-based: templates + randomization from real seed data. No external API.

### 3. Training Data Formatter (`format_training.py`)

Converts synthetic examples into prompt/response JSONL pairs:

```
Prompt:  "Normalize this bank transaction: Data: 15/03/2024, Descrição: UBER *TRIP SP, Valor: -25.90"
Response: {"date": "2024-03-15", "merchant": "Uber", "description": "UBER *TRIP SP", "amount": -25.90, "category": "Transport"}
```

### 4. Fine-tuning Script (`fine_tune.py`)

- Base model: TinyLlama 1.1B
- QLoRA: 4-bit quantization + LoRA adapters
- Target hardware: RTX 4060 (8GB VRAM)
- Train 3-5 epochs on synthetic dataset
- Monitor: loss, exact match accuracy per field

### 5. Model Inference (`inference.py`)

Handles model loading, adapter merging, and inference configuration:

- Loads base TinyLlama + QLoRA adapters
- Inference config: temperature=0.1, max_new_tokens=256 (deterministic output)
- Exposes a simple `normalize_row(raw_text: str) -> dict` interface

### 6. Normalization Pipeline (`normalize.py`)

- Reads any bank CSV (auto-detects delimiter and encoding via `csv.Sniffer` + `chardet`)
- Code pre-processes amounts (strips R$, converts Brazilian number format, applies sign convention)
- Formats each row as a text prompt using the shared prompt template
- Calls `inference.normalize_row()` for each row
- Validates JSON output against Pydantic schema
- On validation failure: retries once, then logs to `errors.jsonl` and skips
- Collects results and saves as JSON or CSV
- Prints summary: total rows, successful, failed

### 7. Evaluation (`evaluate.py`)

- Held-out real CSVs (never seen during training)
- Metrics per field: date exact match, merchant exact match (lowercased), category accuracy + confusion matrix, amount exact match
- Optional: compare against large model (Claude/GPT-4) for accuracy, cost, latency

### 8. API (`api.py`)

- FastAPI POST `/normalize` endpoint
- Accepts CSV file upload (max 10MB)
- Returns normalized JSON array
- No authentication (local tool)

## Data Flow

```
Real CSVs → labeled examples → synthetic generator → training JSONL
                                                          ↓
                                                    fine-tune TinyLlama
                                                          ↓
New CSV → normalize.py reads rows → model normalizes each row → schema validation → output JSON/CSV
```

## Tech Stack

- Python 3.11+
- PyTorch, Transformers, PEFT, BitsAndBytes, TRL (QLoRA fine-tuning)
- Pandas (CSV handling)
- Pydantic (schema validation)
- FastAPI + Uvicorn (API serving)
- pytest (testing)

## File Structure

```
csv-normalizer-model/
├── pyproject.toml
├── .gitignore
├── src/
│   └── csv_normalizer/
│       ├── __init__.py
│       ├── schema.py
│       ├── generate_synthetic.py
│       ├── format_training.py
│       ├── fine_tune.py
│       ├── inference.py
│       ├── normalize.py
│       ├── evaluate.py
│       └── api.py
├── tests/
│   ├── __init__.py
│   ├── test_schema.py
│   ├── test_generate_synthetic.py
│   ├── test_format_training.py
│   ├── test_inference.py
│   ├── test_normalize.py
│   ├── test_evaluate.py
│   └── test_api.py
├── data/
│   ├── raw/           (gitignored — real bank CSVs)
│   ├── labeled/       (manually labeled examples)
│   ├── synthetic/     (generated training data)
│   └── test/          (held-out test CSVs)
└── docs/
    └── superpowers/
        ├── specs/
        └── plans/
```

## Decisions

- **TinyLlama 1.1B** over LLaMA 3.2 1B — smaller, well-supported, fast iteration
- **Rule-based synthetic generation** — real CSVs provide authentic data, templates + randomization gives enough variety
- **No API auth** — local development tool
- **Target: RTX 4060** (8GB VRAM) — QLoRA with TinyLlama fits comfortably
