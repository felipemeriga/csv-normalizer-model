"""FastAPI server for CSV normalization.

POST /normalize: Upload a CSV file, get back normalized JSON.
GET /health: Health check.

Usage: uvicorn csv_normalizer.api:app --host 0.0.0.0 --port 8000
"""

import logging
import os
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from csv_normalizer.csv_detect import read_csv_auto
from csv_normalizer.inference import load_model
from csv_normalizer.label_helper import build_raw_text
from csv_normalizer.normalize import _process_row

logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

app = FastAPI(title="CSV Bank Transaction Normalizer")

# Load model at startup
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "output/adapter")
_model = None
_tokenizer = None


def _get_model():
    global _model, _tokenizer
    if _model is None:
        _model, _tokenizer = load_model(ADAPTER_PATH)
    return _model, _tokenizer


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/normalize")
async def normalize(file: UploadFile = File(...)):
    # Check file size
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large (max 10MB)")

    # Write to temp file for csv_detect to read
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        rows = read_csv_auto(tmp_path)
    finally:
        os.unlink(tmp_path)

    model, tokenizer = _get_model()
    results = []
    errors = []

    for i, row in enumerate(rows):
        result = _process_row(row, model, tokenizer)
        if result is not None:
            results.append(result.model_dump())
        else:
            errors.append({"row_index": i, "raw_text": build_raw_text(row)})

    stats = {
        "total": len(rows),
        "success": len(results),
        "failed": len(errors),
    }

    return JSONResponse(content={"results": results, "stats": stats, "errors": errors})
