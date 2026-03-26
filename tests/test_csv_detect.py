import os
import tempfile

from csv_normalizer.csv_detect import detect_csv_params, read_csv_auto


def test_detect_comma_delimiter():
    content = b"date,description,amount\n2024-01-01,UBER,25.90\n"
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        f.write(content)
        f.flush()
        params = detect_csv_params(f.name)
    os.unlink(f.name)
    assert params["delimiter"] == ","
    assert params["encoding"] == "utf-8"


def test_detect_semicolon_delimiter():
    content = b"data;descricao;valor\n2024-01-01;UBER;25.90\n"
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        f.write(content)
        f.flush()
        params = detect_csv_params(f.name)
    os.unlink(f.name)
    assert params["delimiter"] == ";"


def test_detect_latin1_encoding():
    content = "data;descrição;valor\n2024-01-01;UBER;25.90\n".encode("latin-1")
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        f.write(content)
        f.flush()
        params = detect_csv_params(f.name)
    os.unlink(f.name)
    assert params["encoding"].lower().replace("-", "") in ("latin1", "iso88591", "windows1252")


def test_read_csv_auto_returns_rows():
    content = b"date,description,amount\n2024-01-01,UBER,25.90\n2024-01-02,IFOOD,15.00\n"
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        f.write(content)
        f.flush()
        rows = read_csv_auto(f.name)
    os.unlink(f.name)
    assert len(rows) == 2
    assert rows[0]["description"] == "UBER"


def test_read_csv_auto_semicolon():
    content = b"data;descricao;valor\n2024-01-01;UBER;25.90\n"
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        f.write(content)
        f.flush()
        rows = read_csv_auto(f.name)
    os.unlink(f.name)
    assert len(rows) == 1
    assert rows[0]["descricao"] == "UBER"
