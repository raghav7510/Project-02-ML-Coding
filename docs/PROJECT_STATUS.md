# Project Status

## Completed

- Environment and CUDA stack validated
- RTX 4050 6 GB GPU validated for 4-bit QLoRA
- Qwen3.5-4B-Base loaded successfully in 4-bit NF4
- Baseline generation established
- CodeAlpaca-20k downloaded and audited
- Dataset curated for Python + ML/Data Science coding
- Train/validation/test split created with leakage checks
- Qwen3.5 hybrid architecture inspected
- LoRA target selection validated with `all-linear`
- Trainable parameter count verified
- Forward/backward/optimizer dry run completed
- 512-token VRAM stress test completed
- Final 256-token QLoRA configuration trained successfully
- Final adapter saved locally
- 50-example deterministic generation evaluation completed
- 50-example functional execution evaluation completed

## Final training metrics

| Metric | Value |
|---|---:|
| Training examples | 4,270 |
| Validation examples | 534 |
| Test examples | 534 |
| Epochs | 1 |
| Steps | 534 |
| Training loss | 0.08355 |
| Validation loss | ~0.3977 |
| Validation token accuracy | ~88.14% |
| Runtime | 5,318.73 s |
| Peak allocated VRAM | 5.443 GB |
| Peak reserved VRAM | 5.756 GB |

## Initial held-out evaluation

| Metric | Base | QLoRA |
|---|---:|---:|
| Mean generation time | 62.69 s | 219.82 s |
| Mean generation throughput | 7.89 tok/s | 4.38 tok/s |
| Hit 1,024-token ceiling | 6/50 | 46/50 |
| Functional execution rate | **22.0%** | **88.0%** |

The execution result measures executability rather than semantic correctness.

## Next

- Semantic/task-level correctness evaluation
- Detailed error analysis
- HumanEval/MBPP compatibility check
- Final report and portfolio publication