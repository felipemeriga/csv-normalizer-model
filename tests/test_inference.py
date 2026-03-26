import json
from unittest.mock import MagicMock, patch

from csv_normalizer.inference import normalize_row_with_model, parse_model_output


def test_parse_valid_json():
    raw = (
        '{"date": "2024-03-15", "merchant": "Uber",'
        ' "description": "UBER *TRIP", "amount": -25.9,'
        ' "category": "Transport"}'
    )
    result = parse_model_output(raw)
    assert result["merchant"] == "Uber"
    assert result["category"] == "Transport"


def test_parse_json_with_surrounding_text():
    raw = (
        'Here is the result: {"date": "2024-03-15",'
        ' "merchant": "Uber", "description": "UBER",'
        ' "amount": -25.9, "category": "Transport"} done.'
    )
    result = parse_model_output(raw)
    assert result["merchant"] == "Uber"


def test_parse_invalid_json_returns_none():
    result = parse_model_output("not json at all")
    assert result is None


def test_parse_incomplete_json_returns_none():
    result = parse_model_output('{"date": "2024-03-15"')
    assert result is None


def test_normalize_row_with_model_calls_model():
    expected = {
        "date": "2024-03-15",
        "merchant": "Uber",
        "description": "UBER",
        "amount": -25.9,
        "category": "Transport",
    }

    with patch(
        "csv_normalizer.inference.parse_model_output", return_value=expected
    ):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_input_ids = MagicMock()
        mock_input_ids.shape = [1, 5]
        mock_input_ids.to.return_value = mock_input_ids
        mock_tokenizer.return_value = {"input_ids": mock_input_ids}
        mock_model.generate.return_value = MagicMock()
        mock_model.device = "cpu"
        mock_tokenizer.decode.return_value = json.dumps(expected)

        result = normalize_row_with_model(
            "some raw text", mock_model, mock_tokenizer
        )
    assert result == expected
