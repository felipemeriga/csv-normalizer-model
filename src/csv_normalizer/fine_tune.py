"""QLoRA fine-tuning script for TinyLlama 1.1B.

Trains the model on prompt/response pairs from training.jsonl.
Target hardware: RTX 4060 (8GB VRAM).

Usage: python -m csv_normalizer.fine_tune data/synthetic/training.jsonl --output output/
"""

import json
import sys
from pathlib import Path

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def get_qlora_config() -> dict:
    """Return QLoRA configuration optimized for RTX 4060 (8GB VRAM)."""
    return {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "bits": 4,
        "max_seq_length": 512,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "warmup_ratio": 0.03,
        "logging_steps": 10,
    }


def load_training_data(path: str) -> list[dict]:
    """Load training JSONL file."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def fine_tune(training_path: str, output_dir: str = "output/") -> None:
    """Run QLoRA fine-tuning on TinyLlama.

    Requires: torch, transformers, peft, bitsandbytes, trl.
    Must be run on a machine with a CUDA GPU.
    """
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from trl import SFTTrainer

    config = get_qlora_config()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Load data (messages format)
    data = load_training_data(training_path)
    dataset = Dataset.from_list(data)

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=config["r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output),
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        warmup_ratio=config["warmup_ratio"],
        logging_steps=config["logging_steps"],
        save_strategy="epoch",
        fp16=True,
        report_to="none",
    )

    # Train
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        max_seq_length=config["max_seq_length"],
    )
    trainer.train()

    # Save adapter
    model.save_pretrained(str(output / "adapter"))
    tokenizer.save_pretrained(str(output / "adapter"))
    print(f"Adapter saved to {output / 'adapter'}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m csv_normalizer.fine_tune <training.jsonl> [--output DIR]")
        sys.exit(1)
    training = sys.argv[1]
    out = "output/"
    if "--output" in sys.argv:
        out = sys.argv[sys.argv.index("--output") + 1]
    fine_tune(training, out)
