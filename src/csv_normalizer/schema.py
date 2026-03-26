import re
from enum import Enum

from pydantic import BaseModel, field_validator

PROMPT_TEMPLATE = "Normalize this bank transaction: {raw_row_text}"


class Category(str, Enum):
    TRANSPORT = "Transport"
    FOOD = "Food"
    SHOPPING = "Shopping"
    TRANSFER = "Transfer"
    BILLS = "Bills"
    ENTERTAINMENT = "Entertainment"
    HEALTH = "Health"
    EDUCATION = "Education"
    SUBSCRIPTIONS = "Subscriptions"
    OTHER = "Other"


class NormalizedTransaction(BaseModel):
    date: str
    merchant: str
    description: str
    amount: float
    category: Category

    @field_validator("date")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", v):
            raise ValueError(f"Date must be YYYY-MM-DD format, got: {v}")
        return v
