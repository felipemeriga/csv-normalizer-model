import re


def parse_brazilian_amount(raw: str) -> float:
    """Parse a Brazilian-format amount string into a float.

    Handles: R$ prefix, 1.000,00 format, D/C suffixes, whitespace.
    Convention: negative = debit, positive = credit.
    """
    s = raw.strip()
    if not s:
        raise ValueError("Empty amount string")

    # Strip currency symbol
    s = re.sub(r"R\$\s*", "", s)
    s = s.strip()

    # Check for D/C suffix
    sign = 1.0
    if re.search(r"\s+D$", s, re.IGNORECASE):
        sign = -1.0
        s = re.sub(r"\s+D$", "", s, flags=re.IGNORECASE)
    elif re.search(r"\s+C$", s, re.IGNORECASE):
        sign = 1.0
        s = re.sub(r"\s+C$", "", s, flags=re.IGNORECASE)

    s = s.strip()

    # Detect Brazilian format: dots as thousands, comma as decimal
    if "," in s:
        s = s.replace(".", "")
        s = s.replace(",", ".")

    try:
        return sign * float(s)
    except ValueError:
        raise ValueError(f"Cannot parse amount: {raw!r}")
