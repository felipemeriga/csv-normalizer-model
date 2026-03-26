"""Interactive labeling helper for bank CSV transactions.

Reads real bank CSVs, pre-fills what it can guess, and lets the user
confirm/correct each row. Outputs labeled.jsonl for synthetic data generation.

Usage: python -m csv_normalizer.label_helper data/raw/nubank.csv
"""

import json
import re
import sys
import unicodedata
from pathlib import Path

from csv_normalizer.amount import parse_brazilian_amount
from csv_normalizer.csv_detect import read_csv_auto
from csv_normalizer.schema import Category

LABELED_PATH = Path("data/labeled/labeled.jsonl")
CATEGORIES = [c.value for c in Category]


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
        if _strip_accents(key.lower().strip()) in normalized_candidates:
            return row[key]
    return None


def _prompt_category() -> str:
    """Show category menu and get user choice."""
    print("\nCategories:")
    for i, cat in enumerate(CATEGORIES, 1):
        print(f"  {i}. {cat}")
    while True:
        choice = input("Category (number or name): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(CATEGORIES):
            return CATEGORIES[int(choice) - 1]
        for cat in CATEGORIES:
            if cat.lower() == choice.lower():
                return cat
        print("Invalid choice, try again.")


def label_file(csv_path: str, output_path: str | Path = LABELED_PATH) -> None:
    """Interactively label transactions from a CSV file."""
    rows = read_csv_auto(csv_path)
    skip = load_progress(output_path)

    if skip > 0:
        print(f"Resuming from row {skip + 1} ({skip} already labeled)")

    date_candidates = [
        "data",
        "date",
        "data transacao",
        "data compra",
    ]
    desc_candidates = [
        "title",
        "titulo",
        "descricao",
        "description",
        "estabelecimento",
    ]
    amount_candidates = ["valor", "amount", "value", "quantia"]

    for i, row in enumerate(rows):
        if i < skip:
            continue

        raw_text = build_raw_text(row)
        print(f"\n--- Row {i + 1}/{len(rows)} ---")
        print(f"Raw: {raw_text}")

        # Pre-fill date
        raw_date = _find_column(row, date_candidates) or ""
        guessed_date = guess_date(raw_date) if raw_date else None
        if guessed_date:
            date_input = input(f"Date [{guessed_date}]: ").strip() or guessed_date
        else:
            date_input = input("Date (YYYY-MM-DD): ").strip()

        # Pre-fill description
        raw_desc = _find_column(row, desc_candidates) or ""
        print(f"Description: {raw_desc}")

        # Merchant
        merchant = input("Merchant name: ").strip()

        # Category
        category = _prompt_category()

        # Amount
        raw_amount = _find_column(row, amount_candidates) or ""
        try:
            guessed_amount = parse_brazilian_amount(raw_amount) if raw_amount else None
        except ValueError:
            guessed_amount = None

        if guessed_amount is not None:
            amount_input = input(f"Amount [{guessed_amount}]: ").strip()
            amount = float(amount_input) if amount_input else guessed_amount
        else:
            amount = float(input("Amount: ").strip())

        entry = {
            "raw_text": raw_text,
            "date": date_input,
            "merchant": merchant,
            "description": raw_desc,
            "amount": amount,
            "category": category,
        }
        save_labeled_row(output_path, entry)
        print("  Saved")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m csv_normalizer.label_helper <csv_file>")
        sys.exit(1)
    label_file(sys.argv[1])
