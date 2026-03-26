import json
import os
import tempfile

from csv_normalizer.fine_tune import get_qlora_config, load_training_data


def _create_training_file() -> str:
    entries = [
        {
            "prompt": "Normalize this bank transaction: Data: 15/03/2024",
            "response": '{"date":"2024-03-15"}',
        },
        {
            "prompt": "Normalize this bank transaction: Data: 20/03/2024",
            "response": '{"date":"2024-03-20"}',
        },
    ]
    f = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w")
    for e in entries:
        f.write(json.dumps(e) + "\n")
    f.close()
    return f.name


def test_load_training_data():
    path = _create_training_file()
    try:
        data = load_training_data(path)
        assert len(data) == 2
        assert "prompt" in data[0]
        assert "response" in data[0]
    finally:
        os.unlink(path)


def test_qlora_config_has_required_fields():
    config = get_qlora_config()
    assert config["r"] > 0
    assert config["lora_alpha"] > 0
    assert "target_modules" in config
    assert config["bits"] == 4


def test_qlora_config_fits_rtx4060():
    config = get_qlora_config()
    # 4-bit quantization is required for 8GB VRAM
    assert config["bits"] == 4
    # LoRA rank should be reasonable (not too high for memory)
    assert config["r"] <= 64
