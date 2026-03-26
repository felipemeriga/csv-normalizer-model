import json
import os
import tempfile

from csv_normalizer.label_helper import (
    _find_column,
    auto_label_file,
    build_raw_text,
    clean_merchant,
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
            "category": "Other",
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


def test_clean_merchant_strips_prefix():
    assert clean_merchant("UBER *TRIP SP") == "Trip"
    assert clean_merchant("IFOOD *DELIVERY") == "Delivery"
    assert clean_merchant("PAG*Riachuelo") == "Riachuelo"


def test_clean_merchant_strips_city_suffix():
    assert clean_merchant("RESTAURANTE BOM SAO PAULO") == "Restaurante Bom"


def test_clean_merchant_title_case_all_caps():
    assert clean_merchant("MERCADO LIVRE") == "Mercado Livre"


def test_clean_merchant_preserves_mixed_case():
    assert clean_merchant("iFood") == "iFood"


def test_clean_merchant_empty():
    assert clean_merchant("") == ""


def test_auto_label_file():
    csv_content = (
        b"date,title,amount\n2024-03-15,UBER *TRIP SP,-25.90\n2024-03-20,IFOOD *PIZZA,-15.00\n"
    )

    csv_file = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    csv_file.write(csv_content)
    csv_file.close()

    out_file = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w")
    out_file.close()

    try:
        stats = auto_label_file(csv_file.name, out_file.name)
        assert stats["labeled"] == 2
        assert stats["skipped"] == 0

        with open(out_file.name) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 2
        assert lines[0]["date"] == "2024-03-15"
        assert lines[0]["description"] == "UBER *TRIP SP"
        assert lines[0]["category"] == "Other"
        assert lines[0]["merchant"] != ""
    finally:
        os.unlink(csv_file.name)
        os.unlink(out_file.name)


def test_auto_label_file_skips_rows_without_date():
    csv_content = b"title,amount\nUBER,-25.90\n"

    csv_file = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    csv_file.write(csv_content)
    csv_file.close()

    out_file = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w")
    out_file.close()

    try:
        stats = auto_label_file(csv_file.name, out_file.name)
        assert stats["labeled"] == 0
        assert stats["skipped"] == 1
    finally:
        os.unlink(csv_file.name)
        os.unlink(out_file.name)


def test_auto_label_file_semicolon_delimiter():
    csv_content = "Data;Descrição;Valor\n01/11/2024;UBER *TRIP;25,50\n".encode("utf-8")

    csv_file = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    csv_file.write(csv_content)
    csv_file.close()

    out_file = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w")
    out_file.close()

    try:
        stats = auto_label_file(csv_file.name, out_file.name)
        assert stats["labeled"] == 1

        with open(out_file.name) as f:
            entry = json.loads(f.readline())
        assert entry["date"] == "2024-11-01"
        assert entry["amount"] == 25.50
    finally:
        os.unlink(csv_file.name)
        os.unlink(out_file.name)
