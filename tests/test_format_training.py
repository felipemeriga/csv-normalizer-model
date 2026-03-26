import json
import os
import tempfile

from csv_normalizer.format_training import format_for_training
from csv_normalizer.schema import PROMPT_TEMPLATE


def _create_synthetic_file() -> str:
    entries = [
        {
            "raw_text": "Data: 15/03/2024, Descricao: UBER *TRIP SP, Valor: -25.90",
            "expected": {
                "date": "2024-03-15",
                "merchant": "Uber",
                "description": "UBER *TRIP SP",
                "amount": -25.90,
                "category": "Transport",
            },
        },
    ]
    f = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w")
    for e in entries:
        f.write(json.dumps(e) + "\n")
    f.close()
    return f.name


def test_formats_prompt_response_pairs():
    src = _create_synthetic_file()
    out = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
    out.close()
    try:
        format_for_training(src, out.name)
        with open(out.name) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 1
        assert "prompt" in lines[0]
        assert "response" in lines[0]
    finally:
        os.unlink(src)
        os.unlink(out.name)


def test_prompt_uses_shared_template():
    src = _create_synthetic_file()
    out = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
    out.close()
    try:
        format_for_training(src, out.name)
        with open(out.name) as f:
            line = json.loads(f.readline())
        expected_prompt = PROMPT_TEMPLATE.format(
            raw_row_text="Data: 15/03/2024, Descricao: UBER *TRIP SP, Valor: -25.90"
        )
        assert line["prompt"] == expected_prompt
    finally:
        os.unlink(src)
        os.unlink(out.name)


def test_response_is_valid_json():
    src = _create_synthetic_file()
    out = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
    out.close()
    try:
        format_for_training(src, out.name)
        with open(out.name) as f:
            line = json.loads(f.readline())
        response = json.loads(line["response"])
        assert response["merchant"] == "Uber"
        assert response["category"] == "Transport"
    finally:
        os.unlink(src)
        os.unlink(out.name)
