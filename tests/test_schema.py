import pytest

from csv_normalizer.schema import PROMPT_TEMPLATE, Category, NormalizedTransaction


def test_category_enum_has_10_values():
    assert len(Category) == 10


def test_category_enum_values():
    expected = {
        "Transport",
        "Food",
        "Shopping",
        "Transfer",
        "Bills",
        "Entertainment",
        "Health",
        "Education",
        "Subscriptions",
        "Other",
    }
    assert {c.value for c in Category} == expected


def test_valid_transaction():
    t = NormalizedTransaction(
        date="2024-03-15",
        merchant="Uber",
        description="UBER *TRIP SP",
        amount=-25.90,
        category=Category.TRANSPORT,
    )
    assert t.date == "2024-03-15"
    assert t.merchant == "Uber"
    assert t.amount == -25.90


def test_invalid_date_format_rejected():
    with pytest.raises(Exception):
        NormalizedTransaction(
            date="15/03/2024",
            merchant="Uber",
            description="UBER *TRIP SP",
            amount=-25.90,
            category=Category.TRANSPORT,
        )


def test_invalid_category_rejected():
    with pytest.raises(Exception):
        NormalizedTransaction(
            date="2024-03-15",
            merchant="Uber",
            description="UBER *TRIP SP",
            amount=-25.90,
            category="InvalidCategory",
        )


def test_prompt_template_contains_placeholder():
    assert "{raw_row_text}" in PROMPT_TEMPLATE


def test_transaction_to_dict():
    t = NormalizedTransaction(
        date="2024-03-15",
        merchant="Uber",
        description="UBER *TRIP SP",
        amount=-25.90,
        category=Category.TRANSPORT,
    )
    d = t.model_dump()
    assert d["category"] == "Transport"
    assert isinstance(d["amount"], float)
