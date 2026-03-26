import json
import os
import tempfile

from csv_normalizer.label_helper import (
    _find_column,
    build_raw_text,
    guess_date,
    load_progress,
    save_labeled_row,
)


def test_guess_date_dd_mm_yyyy_slash():
    assert guess_date("15/03/2024") == "2024-03-15"


def test_guess_date_yyyy_mm_dd():
    assert guess_date("2024-03-15") == "2024-03-15"


def test_guess_date_dd_mm_yyyy_dash():
    assert guess_date("15-03-2024") == "2024-03-15"


def test_guess_date_unknown_returns_none():
    assert guess_date("garbage") is None


def test_build_raw_text():
    row = {"Data": "15/03/2024", "Descricao": "UBER *TRIP", "Valor": "-25,90"}
    raw = build_raw_text(row)
    assert "Data: 15/03/2024" in raw
    assert "Descricao: UBER *TRIP" in raw
    assert "Valor: -25,90" in raw


def test_save_labeled_row_creates_jsonl():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        path = f.name

    try:
        entry = {
            "raw_text": "Data: 15/03/2024, Descricao: UBER *TRIP, Valor: -25.90",
            "date": "2024-03-15",
            "merchant": "Uber",
            "description": "UBER *TRIP",
            "amount": -25.90,
            "category": "Transport",
        }
        save_labeled_row(path, entry)
        save_labeled_row(path, entry)

        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 2
        parsed = json.loads(lines[0])
        assert parsed["merchant"] == "Uber"
    finally:
        os.unlink(path)


def test_load_progress_empty_file():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        path = f.name
    try:
        assert load_progress(path) == 0
    finally:
        os.unlink(path)


def test_load_progress_counts_lines():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        f.write('{"a":1}\n{"b":2}\n')
        path = f.name
    try:
        assert load_progress(path) == 2
    finally:
        os.unlink(path)


def test_find_column_with_accents():
    row = {"Título": "UBER *TRIP SP", "Valor": "-25,90"}
    assert _find_column(row, ["titulo"]) == "UBER *TRIP SP"


def test_find_column_case_insensitive():
    row = {"DESCRICAO": "UBER *TRIP SP"}
    assert _find_column(row, ["descricao"]) == "UBER *TRIP SP"


def test_find_column_nubank_title():
    row = {"title": "Uber *Trip", "amount": "-25.90"}
    assert _find_column(row, ["title", "descricao"]) == "Uber *Trip"
