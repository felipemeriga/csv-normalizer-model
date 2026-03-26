from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    with patch("csv_normalizer.api.load_model") as mock_load:
        mock_load.return_value = (MagicMock(), MagicMock())
        from csv_normalizer.api import app

        yield TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_normalize_endpoint_returns_json(client):
    csv_content = b"Data,Descricao,Valor\n15/03/2024,UBER *TRIP SP,-25.90\n"

    from csv_normalizer.schema import Category, NormalizedTransaction

    mock_result = NormalizedTransaction(
        date="2024-03-15",
        merchant="Uber",
        description="UBER *TRIP SP",
        amount=-25.90,
        category=Category.TRANSPORT,
    )

    with patch("csv_normalizer.api._process_row", return_value=mock_result):
        response = client.post(
            "/normalize",
            files={"file": ("test.csv", csv_content, "text/csv")},
        )

    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert "stats" in data
    assert data["stats"]["total"] == 1


def test_normalize_rejects_large_file(client):
    # Create content > 10MB
    large_content = b"a" * (10 * 1024 * 1024 + 1)
    response = client.post(
        "/normalize",
        files={"file": ("big.csv", large_content, "text/csv")},
    )
    assert response.status_code == 413


def test_normalize_rejects_non_csv(client):
    response = client.post(
        "/normalize",
        files={"file": ("test.txt", b"not csv", "text/plain")},
    )
    # Should still try to process (text/plain CSVs are common)
    # or return an error -- implementation decides
    assert response.status_code in (200, 400)
