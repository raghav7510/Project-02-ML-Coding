# Final Evaluation — Qwen3.5-4B Base vs QLoRA Adapter

## Evaluation scope

The fine-tuned adapter was evaluated against the untouched Qwen3.5-4B-Base model on held-out data.

The project held out **534 examples** from the 5,338-example curated dataset. These examples were not used during QLoRA training.

Two complementary 50-example evaluations were completed:

1. **Generation evaluation** — deterministic generation behavior and throughput.
2. **Functional execution evaluation** — whether generated Python could be extracted and executed by the project's evaluator.

The two 50-example evaluations use separate deterministic samples from the held-out set and should not be presented as the exact same 50 prompts.

## 1. Generation evaluation

Configuration:
- Held-out test set: 534 examples
- Evaluation sample: 50 examples
- Deterministic decoding: enabled
- Sampling: disabled
- Thinking mode: disabled
- Maximum generated tokens: 1,024
- Maximum input tokens: 256

### Aggregate generation results

| Metric | Base | QLoRA fine-tuned |
|---|---:|---:|
| Mean generation time | 62.69 s | 219.82 s |
| Median generation time | 63.12 s | 232.32 s |
| Mean generation throughput | 7.89 tok/s | 4.38 tok/s |
| Examples reaching 1,024-token ceiling | 6/50 (12%) | 46/50 (92%) |

The fine-tuned model generated substantially more tokens and frequently reached the 1,024-token ceiling. Therefore generation-time differences should not be interpreted as a pure inference-speed comparison.

## 2. Functional execution evaluation

A separate deterministic sample of 50 held-out examples was passed through the project's execution evaluator.

| Metric | Base | QLoRA fine-tuned |
|---|---:|---:|
| Successfully executed | 11/50 | 44/50 |
| Execution rate | **22.0%** | **88.0%** |

The displayed per-example results showed many base-model failures classified as `syntax_error`, `no_code`, or `no_function`. The fine-tuned model converted many of these into executable outputs.

The fine-tuned sample still contained failures, including syntax errors, a runtime error, a timeout, and a no-code case.

### Interpretation

The execution result is evidence of a substantial improvement in **executability** on this sample.

It is **not equivalent to code correctness**. An executable program can still implement the wrong algorithm, return the wrong value, or fail edge cases.

Therefore the project does not report 88% as a code-accuracy score.

## 3. Training metrics

| Metric | Result |
|---|---:|
| Training examples | 4,270 |
| Validation examples | 534 |
| Epochs | 1 |
| Steps | 534 |
| Training loss | 0.08355 |
| Validation loss | ~0.3977 |
| Validation mean token accuracy | ~88.14% |
| Runtime | 5,318.73 s |
| Peak allocated VRAM | 5.443 GB |
| Peak reserved VRAM | 5.756 GB |

The training/validation loss gap is substantial and is treated as a reason for evaluation rather than as proof of either success or failure.

## 4. Hardware constraint validation

A 512-token memory stress test reached approximately 6.279 GB allocated and 6.604 GB reserved VRAM, exceeding the physical 6 GB GPU capacity.

The final 256-token training configuration completed at 5.443 GB allocated and 5.756 GB reserved VRAM.

## 5. What the evaluation establishes

- Qwen3.5-4B-Base can be adapted with QLoRA on a 6 GB RTX 4050.
- The final adapter uses 32,464,896 trainable parameters, approximately 0.766% of the 4.238B-parameter model.
- One epoch of QLoRA SFT completed successfully.
- On the functional 50-example sample, generated Python became substantially more executable: 22.0% to 88.0%.
- The fine-tuned model produced substantially longer responses under the 1,024-token generation ceiling.

## 6. What the evaluation does not establish

- 88% semantic code correctness.
- Improvement on HumanEval or MBPP.
- General superiority on real-world software engineering.
- Generalization beyond the CodeAlpaca-derived task distribution.
- That longer generated outputs are intrinsically better.

## 7. Next evaluation layer

The next useful layer is semantic correctness: extract executable code safely, run task-specific tests where expected behavior can be derived, compare outputs against reference behavior, and categorize failures by syntax, API usage, algorithm, edge case, and completeness.

Until that layer is completed, the strongest conclusion is that **QLoRA substantially improved executability on the project's held-out functional sample**, while semantic correctness remains to be measured.

## Reproducibility

```text
Model: Qwen/Qwen3.5-4B-Base
Quantization: 4-bit NF4 + double quantization
LoRA: r=16, alpha=32, dropout=0.05
Targets: all-linear
Max length: 256
Train batch: 1
Gradient accumulation: 8
Epochs: 1
Learning rate: 1e-4
Optimizer: paged_adamw_8bit
Seed: 42
```