import csv
import logging

import chardet

logger = logging.getLogger(__name__)


def detect_csv_params(file_path: str) -> dict:
    """Detect encoding and delimiter of a CSV file.

    Returns dict with 'encoding' and 'delimiter' keys.
    Falls back to UTF-8 + comma on failure.
    """
    with open(file_path, "rb") as f:
        raw_bytes = f.read()

    # Detect encoding
    detected = chardet.detect(raw_bytes)
    encoding = detected.get("encoding", "utf-8") or "utf-8"
    # ASCII is a subset of UTF-8; normalize for consistency
    if encoding.lower() == "ascii":
        encoding = "utf-8"

    # Detect delimiter
    try:
        text = raw_bytes.decode(encoding)
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(text[:8192])
        delimiter = dialect.delimiter
    except Exception:
        logger.warning("CSV detection failed for %s, falling back to UTF-8 + comma", file_path)
        encoding = "utf-8"
        delimiter = ","

    return {"encoding": encoding, "delimiter": delimiter}


def read_csv_auto(file_path: str) -> list[dict]:
    """Read a CSV file with auto-detected encoding and delimiter.

    Returns a list of dicts (one per row), using the header as keys.
    """
    params = detect_csv_params(file_path)
    with open(file_path, encoding=params["encoding"], newline="") as f:
        reader = csv.DictReader(f, delimiter=params["delimiter"])
        return list(reader)
