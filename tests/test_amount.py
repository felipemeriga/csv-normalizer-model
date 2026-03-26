import pytest

from csv_normalizer.amount import parse_brazilian_amount


def test_simple_negative():
    assert parse_brazilian_amount("-25.90") == -25.90


def test_brazilian_format_with_comma():
    assert parse_brazilian_amount("1.259,90") == 1259.90


def test_brazilian_format_negative():
    assert parse_brazilian_amount("-1.259,90") == -1259.90


def test_strip_currency_symbol():
    assert parse_brazilian_amount("R$ 1.259,90") == 1259.90


def test_strip_currency_symbol_no_space():
    assert parse_brazilian_amount("R$1.259,90") == 1259.90


def test_simple_integer():
    assert parse_brazilian_amount("100") == 100.0


def test_whitespace():
    assert parse_brazilian_amount("  R$ 25,50  ") == 25.50


def test_debit_credit_suffix_d():
    assert parse_brazilian_amount("25,90 D") == -25.90


def test_debit_credit_suffix_c():
    assert parse_brazilian_amount("25,90 C") == 25.90


def test_empty_string_raises():
    with pytest.raises(ValueError):
        parse_brazilian_amount("")


def test_non_numeric_raises():
    with pytest.raises(ValueError):
        parse_brazilian_amount("abc")
