import json
import os
import re
import tempfile

from csv_normalizer.generate_synthetic import generate_synthetic_data
from csv_normalizer.schema import Category


def _create_seed_file() -> str:
    """Create a minimal labeled.jsonl seed file."""
    entries = [
        {
            "raw_text": "Data: 15/03/2024, Descrição: UBER *TRIP SP, Valor: -25.90",
            "date": "2024-03-15",
            "merchant": "Uber",
            "description": "UBER *TRIP SP",
            "amount": -25.90,
            "category": "Transport",
        },
        {
            "raw_text": "Data: 20/03/2024, Descrição: RCHLO*Riachuelo, Valor: -159.90",
            "date": "2024-03-20",
            "merchant": "Riachuelo",
            "description": "RCHLO*Riachuelo",
            "amount": -159.90,
            "category": "Shopping",
        },
    ]
    f = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w")
    for e in entries:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")
    f.close()
    return f.name


def test_generates_correct_count():
    seed_path = _create_seed_file()
    out = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
    out.close()
    try:
        generate_synthetic_data(seed_path, out.name, count=50)
        with open(out.name) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 50
    finally:
        os.unlink(seed_path)
        os.unlink(out.name)


def test_output_has_messages_format():
    seed_path = _create_seed_file()
    out = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
    out.close()
    try:
        generate_synthetic_data(seed_path, out.name, count=10)
        with open(out.name) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        for entry in lines:
            assert "messages" in entry
            messages = entry["messages"]
            assert len(messages) == 2
            assert messages[0]["role"] == "user"
            assert messages[1]["role"] == "assistant"
    finally:
        os.unlink(seed_path)
        os.unlink(out.name)


def test_user_message_uses_prompt_template():
    seed_path = _create_seed_file()
    out = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
    out.close()
    try:
        generate_synthetic_data(seed_path, out.name, count=10)
        with open(out.name) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        for entry in lines:
            content = entry["messages"][0]["content"]
            assert content.startswith("Normalize this bank transaction:")
    finally:
        os.unlink(seed_path)
        os.unlink(out.name)


def test_assistant_message_is_valid_json_with_fields():
    seed_path = _create_seed_file()
    out = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
    out.close()
    try:
        generate_synthetic_data(seed_path, out.name, count=10)
        with open(out.name) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        for entry in lines:
            response = json.loads(entry["messages"][1]["content"])
            assert "date" in response
            assert "merchant" in response
            assert "description" in response
            assert "amount" in response
            assert "category" in response
    finally:
        os.unlink(seed_path)
        os.unlink(out.name)


def test_categories_are_valid():
    valid = {c.value for c in Category}
    seed_path = _create_seed_file()
    out = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
    out.close()
    try:
        generate_synthetic_data(seed_path, out.name, count=20)
        with open(out.name) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        for entry in lines:
            response = json.loads(entry["messages"][1]["content"])
            assert response["category"] in valid
    finally:
        os.unlink(seed_path)
        os.unlink(out.name)


def test_dates_are_valid_format():
    seed_path = _create_seed_file()
    out = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
    out.close()
    try:
        generate_synthetic_data(seed_path, out.name, count=20)
        with open(out.name) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        for entry in lines:
            response = json.loads(entry["messages"][1]["content"])
            assert re.match(r"^\d{4}-\d{2}-\d{2}$", response["date"])
    finally:
        os.unlink(seed_path)
        os.unlink(out.name)


def test_raw_text_varies_delimiters():
    seed_path = _create_seed_file()
    out = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
    out.close()
    try:
        generate_synthetic_data(seed_path, out.name, count=100)
        with open(out.name) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        raw_texts = [e["messages"][0]["content"] for e in lines]
        has_comma = any(", " in r for r in raw_texts)
        has_semicolon = any("; " in r for r in raw_texts)
        assert has_comma or has_semicolon
    finally:
        os.unlink(seed_path)
        os.unlink(out.name)
