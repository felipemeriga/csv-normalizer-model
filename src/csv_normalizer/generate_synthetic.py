"""Synthetic training data generator.

Takes labeled.jsonl as seed data and generates variations by randomizing
dates, amounts, column names, delimiters, date formats, and adding noise.

Usage:
    python -m csv_normalizer.generate_synthetic seed.jsonl output.jsonl --count 2000
"""

import json
import random
import sys
from pathlib import Path

# Column name variations for Brazilian banks
COLUMN_NAMES = {
    "date": [
        "Data",
        "data",
        "Date",
        "DATA",
        "Data Transacao",
        "Data Compra",
        "data_transacao",
    ],
    "description": [
        "Descricao",
        "descricao",
        "Description",
        "Estabelecimento",
        "DESCRICAO",
        "Titulo",
        "Lancamento",
    ],
    "amount": [
        "Valor",
        "valor",
        "Amount",
        "VALOR",
        "Quantia",
        "Value",
        "Montante",
    ],
}

# Date format variations
DATE_FORMATS = [
    "{d:02d}/{m:02d}/{y}",  # DD/MM/YYYY
    "{d:02d}-{m:02d}-{y}",  # DD-MM-YYYY
    "{y}-{m:02d}-{d:02d}",  # YYYY-MM-DD
    "{d:02d}/{m:02d}/{sy}",  # DD/MM/YY
    "{d}/{m}/{y}",  # D/M/YYYY (no zero-pad)
]

DELIMITERS = [", ", "; ", " | ", " - "]


def _load_seed(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _random_date() -> tuple[str, str]:
    """Generate a random date, returning (formatted_raw, normalized YYYY-MM-DD)."""
    y = random.randint(2020, 2026)
    m = random.randint(1, 12)
    d = random.randint(1, 28)  # safe for all months
    normalized = f"{y}-{m:02d}-{d:02d}"
    sy = str(y)[2:]  # short year
    fmt = random.choice(DATE_FORMATS)
    raw = fmt.format(d=d, m=m, y=y, sy=sy)
    return raw, normalized


def _random_amount(base: float) -> tuple[str, float]:
    """Generate a random amount variation, returning (raw_str, normalized_float)."""
    factor = random.uniform(0.5, 1.5)
    amount = round(base * factor, 2)

    fmt = random.choice(["plain", "brazilian", "with_symbol"])
    if fmt == "plain":
        raw = f"{amount}"
    elif fmt == "brazilian":
        abs_amount = abs(amount)
        integer_part = int(abs_amount)
        decimal_part = round((abs_amount - integer_part) * 100)
        if integer_part >= 1000:
            int_str = f"{integer_part:,}".replace(",", ".")
        else:
            int_str = str(integer_part)
        raw = f"{'-' if amount < 0 else ''}{int_str},{decimal_part:02d}"
    else:
        raw = f"R$ {abs(amount):.2f}"
        if amount < 0:
            amount = -abs(amount)
            raw = f"-{raw}"

    return raw, amount


def _add_noise(text: str) -> str:
    """Randomly add whitespace noise to text."""
    if random.random() < 0.3:
        text = "  " + text
    if random.random() < 0.3:
        text = text + "  "
    if random.random() < 0.1:
        text = text.replace(": ", ":  ")
    return text


def generate_synthetic_data(seed_path: str, output_path: str, count: int = 2000) -> None:
    """Generate synthetic training data from seed labeled data."""
    seeds = _load_seed(seed_path)
    if not seeds:
        raise ValueError("Seed file is empty")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w") as f:
        for _ in range(count):
            seed = random.choice(seeds)

            raw_date, norm_date = _random_date()
            raw_amount, norm_amount = _random_amount(seed["amount"])

            date_col = random.choice(COLUMN_NAMES["date"])
            desc_col = random.choice(COLUMN_NAMES["description"])
            amount_col = random.choice(COLUMN_NAMES["amount"])

            delim = random.choice(DELIMITERS)

            parts = [
                (date_col, raw_date),
                (desc_col, seed["description"]),
                (amount_col, raw_amount),
            ]
            random.shuffle(parts)
            raw_text = delim.join(f"{k}: {v}" for k, v in parts)
            raw_text = _add_noise(raw_text)

            entry = {
                "raw_text": raw_text,
                "expected": {
                    "date": norm_date,
                    "merchant": seed["merchant"],
                    "description": seed["description"],
                    "amount": norm_amount,
                    "category": seed["category"],
                },
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: python -m csv_normalizer.generate_synthetic"
            " <seed.jsonl> <output.jsonl> [--count N]"
        )
        sys.exit(1)
    seed = sys.argv[1]
    out = sys.argv[2]
    n = 2000
    if "--count" in sys.argv:
        n = int(sys.argv[sys.argv.index("--count") + 1])
    generate_synthetic_data(seed, out, count=n)
