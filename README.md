# Project 02 — ML Coding

Fine-tuning **Qwen3.5-4B-Base** for Python and ML/Data Science coding using **QLoRA** on an RTX 4050 Laptop GPU (6 GB VRAM).

## Objective

Build and evaluate a parameter-efficient fine-tuned coding model that can improve on the base model for Python and ML/Data Science coding tasks while remaining runnable on consumer hardware.

This project is part of a **7 Weeks → 7 AI Projects** portfolio challenge.

## Model

- Base model: `Qwen/Qwen3.5-4B-Base`
- Fine-tuning: QLoRA / LoRA
- Quantization: 4-bit NF4
- Double quantization: enabled
- Compute dtype: FP16
- LoRA rank: 16
- LoRA alpha: 32
- LoRA dropout: 0.05
- Target modules: `all-linear`
- Trainable parameters: 32,464,896
- Total parameters: 4,238,216,192
- Trainable fraction: 0.766%

## Dataset

Source: `sahil2801/CodeAlpaca-20k`

The final curated dataset contains **5,338 examples**:

- 5,000 general Python examples
- 338 ML/Data Science examples
- 4,270 training examples
- 534 validation examples
- 534 held-out test examples
- Random seed: 42
- No prompt/input overlap across train, validation, and test splits

The dataset is converted to conversational `messages` format and uses the Qwen chat template during training/evaluation.

## Training

Hardware:

- NVIDIA GeForce RTX 4050 Laptop GPU
- 6 GB VRAM
- 16 GB system RAM

Final training configuration:

- Sequence length: 256 tokens
- Per-device batch size: 1
- Gradient accumulation: 8
- Effective batch size: 8
- Epochs: 1
- Learning rate: 1e-4
- Weight decay: 0.01
- Warmup steps: 16
- Scheduler: cosine
- Optimizer: paged AdamW 8-bit
- Gradient checkpointing: enabled
- Seed: 42

### Final training result

- Training examples processed: 4,270
- Training steps: 534
- Runtime: 5,318.73 seconds
- Training loss: 0.08355
- Validation loss: ~0.3977
- Validation mean token accuracy: ~0.8814
- Peak allocated VRAM: 5.443 GB
- Peak reserved VRAM: 5.756 GB

The large train/evaluation loss gap is retained as an explicit evaluation point rather than being presented as evidence of successful generalization by itself.

## Engineering constraints

A 512-token memory stress test reached approximately:

- 6.279 GB allocated VRAM
- 6.604 GB reserved VRAM

Therefore, 512 tokens was not considered safe for the 6 GB GPU. The final run used 256 tokens and completed successfully.

## Evaluation plan

The evaluation stage will compare the base model and the fine-tuned adapter using identical prompts and generation settings.

Planned evaluation:

1. Held-out test-set evaluation
2. Baseline vs fine-tuned generation comparison
3. Code execution checks
4. Execution-based coding benchmark(s), where compatible
5. Error analysis and qualitative examples
6. Resource/performance comparison

The held-out test set was not used during fine-tuning.

## Repository structure

```
Project-02-ML-Coding/
├── data/
├── experiments/
├── model/
├── docs/
├── README.md
└── .gitignore
```

Large datasets, model weights, checkpoints, caches, and local virtual environments are intentionally excluded from Git.

## Status

**Fine-tuning: COMPLETE**

**Evaluation: NEXT STAGE**

The final adapter is stored locally under:

```
model/qwen3.5-4b-python-qlora
```

This repository tracks the reproducible project methodology and evaluation artifacts rather than committing large model/dataset binaries.
