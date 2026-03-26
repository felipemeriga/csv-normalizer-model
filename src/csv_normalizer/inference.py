"""Model inference for transaction normalization.

Loads TinyLlama + QLoRA adapters and provides a simple interface
to normalize one transaction row at a time.

Usage:
    from csv_normalizer.inference import load_model, normalize_row_with_model
    model, tokenizer = load_model("output/adapter")
    result = normalize_row_with_model(raw_text, model, tokenizer)
"""

import json
import re

from csv_normalizer.schema import PROMPT_TEMPLATE

INFERENCE_CONFIG = {
    "temperature": 0.1,
    "max_new_tokens": 256,
    "do_sample": True,
}


def parse_model_output(raw_output: str) -> dict | None:
    """Extract a JSON object from model output text.

    The model may include extra text around the JSON. This function
    finds the first valid JSON object in the output.
    Returns None if no valid JSON found.
    """
    matches = re.findall(r"\{[^{}]*\}", raw_output)
    for match in matches:
        try:
            parsed = json.loads(match)
            if "date" in parsed and "merchant" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue
    return None


def normalize_row_with_model(raw_text: str, model, tokenizer) -> dict | None:
    """Normalize a single transaction row using the loaded model.

    Args:
        raw_text: The raw CSV row text (key-value pairs).
        model: The loaded model (base + adapters).
        tokenizer: The loaded tokenizer.

    Returns:
        Parsed dict with normalized fields, or None if parsing fails.
    """
    prompt = PROMPT_TEMPLATE.format(raw_row_text=raw_text)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)

    outputs = model.generate(
        input_ids,
        max_new_tokens=INFERENCE_CONFIG["max_new_tokens"],
        temperature=INFERENCE_CONFIG["temperature"],
        do_sample=INFERENCE_CONFIG["do_sample"],
    )

    # Decode only the generated tokens (skip the prompt)
    generated = outputs[0][input_ids.shape[1] :]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return parse_model_output(text)


def load_model(
    adapter_path: str,
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
):
    """Load base model with QLoRA adapters for inference.

    Returns (model, tokenizer) tuple.
    Must be run on a machine with a CUDA GPU.
    """
    import torch
    from peft import PeftModel
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    return model, tokenizer
