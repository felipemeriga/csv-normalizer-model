"""Format synthetic data into prompt/response pairs for fine-tuning.

Input: synthetic.jsonl (raw_text + expected)
Output: training.jsonl (prompt + response)

Usage:
    python -m csv_normalizer.format_training <input.jsonl> <output.jsonl>
"""

import json
import sys
from pathlib import Path

from csv_normalizer.schema import PROMPT_TEMPLATE


def format_for_training(input_path: str, output_path: str) -> None:
    """Convert synthetic JSONL into prompt/response training pairs."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path) as fin, open(output, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            entry = json.loads(line)
            prompt = PROMPT_TEMPLATE.format(raw_row_text=entry["raw_text"])
            response = json.dumps(entry["expected"], ensure_ascii=False)
            fout.write(json.dumps({"prompt": prompt, "response": response}) + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python -m csv_normalizer.format_training <input.jsonl> <output.jsonl>")
        sys.exit(1)
    format_for_training(sys.argv[1], sys.argv[2])
