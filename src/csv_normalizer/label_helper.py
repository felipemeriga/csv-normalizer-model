"""Automatic labeling helper for bank CSV transactions.

Reads all CSVs from data/raw/, auto-detects fields, cleans merchant names,
and outputs labeled.jsonl for synthetic data generation. No manual input needed.

Usage: python -m csv_normalizer.label_helper [data/raw/]
"""

import json
import logging
import re
import sys
import unicodedata
from pathlib import Path

from csv_normalizer.amount import parse_brazilian_amount
from csv_normalizer.csv_detect import read_csv_auto

logger = logging.getLogger(__name__)

LABELED_PATH = Path("data/labeled/labeled.jsonl")
RAW_DIR = Path("data/raw")

DATE_CANDIDATES = [
    "date",
    "data",
    "data da compra",
    "data transacao",
    "data compra",
    "data de compra",
]
DESC_CANDIDATES = [
    "title",
    "titulo",
    "descricao",
    "description",
    "estabelecimento",
    "lancamento",
    "historico",
]
AMOUNT_CANDIDATES = [
    "amount",
    "valor",
    "valor (em r$)",
    "valor (r$)",
    "value",
    "quantia",
]

# Common prefixes in Brazilian bank descriptions to strip for merchant name
MERCHANT_PREFIXES = re.compile(
    r"^(PAG\*|PAGSEGURO\*|PAG \*|RCHLO\*|UBER \*|UBER\*|"
    r"IFOOD \*|IFOOD\*|EBN \*|Ebn \*|MP \*|"
    r"RAPPI\*|RAPPI \*|MERCPAGO\*|MERCPAGO \*|"
    r"SQ \*|STONE \*|CIELO \*|GETNET \*|REDE \*|"
    r"PIC PAY\*|PICPAY\*|AME\*|PAYPAL \*)",
    re.IGNORECASE,
)

# Suffixes to strip (city codes, installment info)
MERCHANT_SUFFIXES = re.compile(
    r"\s+(BR|SP|RJ|MG|PR|RS|SC|BA|GO|DF|CE|PE|PA|MA|"
    r"SAO PAULO|RIO DE JANEIRO|CURITIBA|BELO HORIZONTE|"
    r"\d+/\d+|\d{2}/\d{2})\s*$",
    re.IGNORECASE,
)


def guess_date(raw: str) -> str | None:
    """Try to parse a date string into YYYY-MM-DD format."""
    raw = raw.strip()

    # YYYY-MM-DD already
    if re.match(r"^\d{4}-\d{2}-\d{2}$", raw):
        return raw

    # DD/MM/YYYY or DD-MM-YYYY
    m = re.match(r"^(\d{2})[/\-](\d{2})[/\-](\d{4})$", raw)
    if m:
        return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"

    # DD/MM/YY
    m = re.match(r"^(\d{2})[/\-](\d{2})[/\-](\d{2})$", raw)
    if m:
        year = int(m.group(3))
        year = 2000 + year if year < 50 else 1900 + year
        return f"{year}-{m.group(2)}-{m.group(1)}"

    # DD/MM (no year — skip, ambiguous)
    return None


def build_raw_text(row: dict) -> str:
    """Convert a CSV row dict into the raw_text format for the prompt."""
    parts = [f"{k}: {v}" for k, v in row.items() if v]
    return ", ".join(parts)


def save_labeled_row(path: str | Path, entry: dict) -> None:
    """Append a single labeled entry as a JSONL line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_progress(path: str | Path) -> int:
    """Count how many rows have already been labeled."""
    path = Path(path)
    if not path.exists():
        return 0
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def _strip_accents(s: str) -> str:
    """Remove accents from a string for comparison."""
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _find_column(row: dict, candidates: list[str]) -> str | None:
    """Find a column value by trying multiple possible column names.

    Comparison is case-insensitive and accent-insensitive.
    """
    normalized_candidates = [_strip_accents(c.lower().strip()) for c in candidates]
    for key in row:
        if key is None:
            continue
        if _strip_accents(key.lower().strip()) in normalized_candidates:
            return row[key]
    return None


def clean_merchant(raw_desc: str) -> str:
    """Clean a raw bank description into a readable merchant name.

    Strips common payment processor prefixes, city/state suffixes,
    and normalizes whitespace.
    """
    if not raw_desc:
        return ""

    cleaned = raw_desc.strip()
    # Strip common prefixes
    cleaned = MERCHANT_PREFIXES.sub("", cleaned)
    # Strip common suffixes
    cleaned = MERCHANT_SUFFIXES.sub("", cleaned)
    # Normalize whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # Title case if all caps
    if cleaned == cleaned.upper() and len(cleaned) > 3:
        cleaned = cleaned.title()

    return cleaned


def auto_label_file(csv_path: str, output_path: str | Path = LABELED_PATH) -> dict:
    """Automatically label all transactions from a CSV file.

    Returns stats dict with total, labeled, skipped counts.
    """
    try:
        rows = read_csv_auto(csv_path)
    except Exception as e:
        logger.warning("Failed to read %s: %s", csv_path, e)
        return {"total": 0, "labeled": 0, "skipped": 0, "file": csv_path}

    labeled = 0
    skipped = 0

    for row in rows:
        # Find date
        raw_date = _find_column(row, DATE_CANDIDATES) or ""
        date = guess_date(raw_date) if raw_date else None
        if not date:
            skipped += 1
            continue

        # Find description
        raw_desc = _find_column(row, DESC_CANDIDATES) or ""
        if not raw_desc:
            skipped += 1
            continue

        # Find amount
        raw_amount = _find_column(row, AMOUNT_CANDIDATES) or ""
        try:
            amount = parse_brazilian_amount(raw_amount) if raw_amount else None
        except ValueError:
            amount = None
        if amount is None:
            skipped += 1
            continue

        # Clean merchant name
        merchant = clean_merchant(raw_desc)

        entry = {
            "raw_text": build_raw_text(row),
            "date": date,
            "merchant": merchant,
            "description": raw_desc,
            "amount": amount,
            "category": "Other",
        }
        save_labeled_row(output_path, entry)
        labeled += 1

    return {"total": len(rows), "labeled": labeled, "skipped": skipped, "file": csv_path}


def auto_label_dir(
    raw_dir: str | Path = RAW_DIR, output_path: str | Path = LABELED_PATH
) -> list[dict]:
    """Automatically label all CSVs in a directory.

    Returns list of stats dicts, one per file.
    """
    raw_dir = Path(raw_dir)
    all_stats = []

    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {raw_dir}")
        return all_stats

    print(f"Found {len(csv_files)} CSV files in {raw_dir}")

    for csv_file in csv_files:
        stats = auto_label_file(str(csv_file), output_path)
        all_stats.append(stats)
        print(f"  {csv_file.name}: {stats['labeled']} labeled, {stats['skipped']} skipped")

    total_labeled = sum(s["labeled"] for s in all_stats)
    total_skipped = sum(s["skipped"] for s in all_stats)
    print(f"\nTotal: {total_labeled} labeled, {total_skipped} skipped from {len(csv_files)} files")

    return all_stats


if __name__ == "__main__":
    raw = sys.argv[1] if len(sys.argv) > 1 else str(RAW_DIR)
    path = Path(raw)
    if path.is_dir():
        auto_label_dir(path)
    else:
        stats = auto_label_file(raw)
        print(f"{stats['labeled']} labeled, {stats['skipped']} skipped")
