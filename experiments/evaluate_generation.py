"""
Project 02 — Base vs QLoRA Evaluation

Runs a controlled generation comparison on the held-out test set.

Both models receive exactly the same prompts, tokenizer/chat template,
decoding configuration, and maximum input/output lengths.

Usage:
    python experiments/evaluate_generation.py --limit 25
    python experiments/evaluate_generation.py --limit 534

Outputs:
    evaluation/results_<limit>.jsonl
    evaluation/summary_<limit>.json
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


MODEL_NAME = "Qwen/Qwen3.5-4B-Base"
ADAPTER_DIR = "model/qwen3.5-4b-python-qlora"
TEST_FILE = "data/test/test.jsonl"
OUTPUT_DIR = "evaluation"

SEED = 42
MAX_INPUT_TOKENS = 256
MAX_NEW_TOKENS = 256
DO_SAMPLE = False
ENABLE_THINKING = False


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_test(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def user_messages(messages):
    return [m for m in messages if m.get("role") in {"system", "user"}]


def build_prompt(tokenizer, messages):
    prompt_messages = user_messages(messages)
    return tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=ENABLE_THINKING,
    )


def generate_one(model, tokenizer, prompt):
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
        add_special_tokens=False,
    )
    encoded = {k: v.to(model.device) for k, v in encoded.items()}
    input_len = encoded["input_ids"].shape[1]

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - start

    generated_ids = output[0, input_len:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    peak_vram = None
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)

    return {
        "output": text,
        "input_tokens": int(input_len),
        "generated_tokens": int(len(generated_ids)),
        "generation_seconds": elapsed,
        "tokens_per_second": (
            len(generated_ids) / elapsed if elapsed > 0 else None
        ),
        "peak_allocated_vram_gb": peak_vram,
    }


def load_base():
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=534)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()

    set_seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this evaluation.")

    if not os.path.isfile(TEST_FILE):
        raise FileNotFoundError(TEST_FILE)

    if not os.path.isdir(ADAPTER_DIR):
        raise FileNotFoundError(
            f"Fine-tuned adapter directory not found: {ADAPTER_DIR}"
        )

    rows = load_test(TEST_FILE)
    end = min(args.start + args.limit, len(rows))
    rows = rows[args.start:end]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    result_path = os.path.join(
        OUTPUT_DIR, f"results_{args.start}_{end}.jsonl"
    )
    summary_path = os.path.join(
        OUTPUT_DIR, f"summary_{args.start}_{end}.json"
    )

    print("=" * 70)
    print("PROJECT 02 — BASE VS QLORA GENERATION EVALUATION")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Adapter: {ADAPTER_DIR}")
    print(f"Test examples: {len(rows)}")
    print(f"Range: {args.start}:{end}")
    print(f"Max input tokens: {MAX_INPUT_TOKENS}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")
    print(f"Sampling: {DO_SAMPLE}")
    print(f"Thinking: {ENABLE_THINKING}")
    print()

    model, tokenizer = load_base()

    records = []
    base_times = []
    ft_times = []
    base_tps = []
    ft_tps = []

    print("PHASE 1 — BASE MODEL")
    print("-" * 70)

    prompts = []
    for idx, row in enumerate(rows, start=args.start):
        messages = row["messages"]
        prompt = build_prompt(tokenizer, messages)
        prompts.append(prompt)

        result = generate_one(model, tokenizer, prompt)
        base_times.append(result["generation_seconds"])
        if result["tokens_per_second"] is not None:
            base_tps.append(result["tokens_per_second"])

        records.append({
            "index": idx,
            "prompt": prompt,
            "reference_answer": messages[-1]["content"],
            "base": result,
        })

        print(
            f"Base {idx + 1 - args.start}/{len(rows)} | "
            f"{result['generation_seconds']:.2f}s | "
            f"{result['generated_tokens']} tokens"
        )

    print()
    print("PHASE 2 — LOAD FINE-TUNED ADAPTER")
    print("-" * 70)

    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    model.eval()

    print("Adapter loaded successfully.")
    print()
    print("PHASE 3 — FINE-TUNED MODEL")
    print("-" * 70)

    for i, record in enumerate(records):
        result = generate_one(model, tokenizer, record["prompt"])
        record["fine_tuned"] = result
        ft_times.append(result["generation_seconds"])
        if result["tokens_per_second"] is not None:
            ft_tps.append(result["tokens_per_second"])

        print(
            f"FT   {i + 1}/{len(records)} | "
            f"{result['generation_seconds']:.2f}s | "
            f"{result['generated_tokens']} tokens"
        )

    with open(result_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "model": MODEL_NAME,
        "adapter": ADAPTER_DIR,
        "test_file": TEST_FILE,
        "seed": SEED,
        "start": args.start,
        "end": end,
        "examples_evaluated": len(records),
        "max_input_tokens": MAX_INPUT_TOKENS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": DO_SAMPLE,
        "enable_thinking": ENABLE_THINKING,
        "base": {
            "mean_generation_seconds": (
                float(np.mean(base_times)) if base_times else None
            ),
            "median_generation_seconds": (
                float(np.median(base_times)) if base_times else None
            ),
            "mean_tokens_per_second": (
                float(np.mean(base_tps)) if base_tps else None
            ),
        },
        "fine_tuned": {
            "mean_generation_seconds": (
                float(np.mean(ft_times)) if ft_times else None
            ),
            "median_generation_seconds": (
                float(np.median(ft_times)) if ft_times else None
            ),
            "mean_tokens_per_second": (
                float(np.mean(ft_tps)) if ft_tps else None
            ),
        },
        "notes": [
            "Both models use identical prompts and deterministic decoding.",
            "Only the user/system portion of each test example is provided to the model.",
            "Reference answers are stored for later qualitative/execution analysis.",
            "The test set was held out during fine-tuning.",
            "This script measures generation behavior; it does not claim code correctness.",
        ],
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Results: {result_path}")
    print(f"Summary: {summary_path}")
    print(f"Examples: {len(records)}")


if __name__ == "__main__":
    main()
