import json
import os
import tempfile
from unittest.mock import MagicMock, patch

from csv_normalizer.normalize import _process_row, normalize_csv


def test_process_row_success():
    row = {"Data": "15/03/2024", "Descricao": "UBER *TRIP SP", "Valor": "-25,90"}
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    expected = {
        "date": "2024-03-15",
        "merchant": "Uber",
        "description": "UBER *TRIP SP",
        "amount": -25.90,
        "category": "Transport",
    }

    with patch("csv_normalizer.normalize.normalize_row_with_model", return_value=expected):
        result = _process_row(row, mock_model, mock_tokenizer)
    assert result is not None
    assert result.merchant == "Uber"


def test_process_row_retry_on_failure():
    row = {"Data": "15/03/2024", "Descricao": "UBER *TRIP SP", "Valor": "-25,90"}
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    expected = {
        "date": "2024-03-15",
        "merchant": "Uber",
        "description": "UBER *TRIP SP",
        "amount": -25.90,
        "category": "Transport",
    }

    with patch("csv_normalizer.normalize.normalize_row_with_model", side_effect=[None, expected]):
        result = _process_row(row, mock_model, mock_tokenizer)
    assert result is not None


def test_process_row_returns_none_after_two_failures():
    row = {"Data": "15/03/2024", "Descricao": "UBER", "Valor": "-25,90"}
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    with patch("csv_normalizer.normalize.normalize_row_with_model", return_value=None):
        result = _process_row(row, mock_model, mock_tokenizer)
    assert result is None


def test_normalize_csv_writes_output():
    csv_content = b"Data,Descricao,Valor\n15/03/2024,UBER,-25.90\n20/03/2024,IFOOD,-15.00\n"

    expected = {
        "date": "2024-03-15",
        "merchant": "Uber",
        "description": "UBER",
        "amount": -25.90,
        "category": "Transport",
    }

    csv_file = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    csv_file.write(csv_content)
    csv_file.close()

    out_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    out_file.close()

    try:
        with patch("csv_normalizer.normalize.normalize_row_with_model", return_value=expected):
            mock_model = MagicMock()
            mock_tokenizer = MagicMock()
            stats = normalize_csv(csv_file.name, out_file.name, mock_model, mock_tokenizer)

        assert stats["total"] == 2
        assert stats["success"] == 2
        assert stats["failed"] == 0

        with open(out_file.name) as f:
            results = json.load(f)
        assert len(results) == 2
    finally:
        os.unlink(csv_file.name)
        os.unlink(out_file.name)
