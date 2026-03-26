"""Normalization pipeline: read CSV -> model -> validated output.

Reads any bank CSV, sends each row to the model for normalization,
validates against the schema, and writes results.

Usage: python -m csv_normalizer.normalize input.csv output.json --adapter output/adapter
"""

import json
import logging
import sys
from pathlib import Path

from csv_normalizer.csv_detect import read_csv_auto
from csv_normalizer.inference import normalize_row_with_model
from csv_normalizer.label_helper import build_raw_text
from csv_normalizer.schema import NormalizedTransaction

logger = logging.getLogger(__name__)

MAX_RETRIES = 2


def _process_row(row: dict, model, tokenizer) -> NormalizedTransaction | None:
    """Process a single CSV row through the model with retry logic.

    Returns a validated NormalizedTransaction or None on failure.
    """
    raw_text = build_raw_text(row)

    for attempt in range(MAX_RETRIES):
        result = normalize_row_with_model(raw_text, model, tokenizer)
        if result is None:
            continue
        try:
            return NormalizedTransaction(**result)
        except Exception:
            continue

    return None


def normalize_csv(
    input_path: str,
    output_path: str,
    model,
    tokenizer,
    errors_path: str | None = None,
) -> dict:
    """Normalize all rows in a CSV file."""
    if errors_path is None:
        errors_path = str(Path(output_path).with_suffix("")) + ".errors.jsonl"

    rows = read_csv_auto(input_path)
    results = []
    errors = []

    for i, row in enumerate(rows):
        result = _process_row(row, model, tokenizer)
        if result is not None:
            results.append(result.model_dump())
        else:
            raw_text = build_raw_text(row)
            error_entry = {"row_index": i, "raw_text": raw_text}
            errors.append(error_entry)
            logger.warning("Failed to normalize row %d: %s", i, raw_text)

    # Write results
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Write errors
    if errors:
        with open(errors_path, "w") as f:
            for e in errors:
                f.write(json.dumps(e) + "\n")

    stats = {
        "total": len(rows),
        "success": len(results),
        "failed": len(errors),
    }
    print(f"Normalized {stats['success']}/{stats['total']} rows ({stats['failed']} failed)")
    return stats


if __name__ == "__main__":
    from csv_normalizer.inference import load_model

    if len(sys.argv) < 3:
        print(
            "Usage: python -m csv_normalizer.normalize"
            " <input.csv> <output.json> --adapter <path>"
        )
        sys.exit(1)

    input_csv = sys.argv[1]
    output_json = sys.argv[2]
    adapter = "output/adapter"
    if "--adapter" in sys.argv:
        adapter = sys.argv[sys.argv.index("--adapter") + 1]

    model, tokenizer = load_model(adapter)
    normalize_csv(input_csv, output_json, model, tokenizer)
