# Project 02 — Detailed Experiment Log

This document preserves the engineering sequence behind the final QLoRA run.

## Phase 1 — Environment

### Working directory

```powershell
C:\Users\ragha\Project02-ML-Coding
```

### Virtual environment

```powershell
.venv\Scripts\Activate.ps1
```

### Hardware

- NVIDIA RTX 4050 Laptop GPU
- 6 GB VRAM
- 16 GB DDR5 system memory

### CUDA stack

- NVIDIA driver: 591.59
- `nvidia-smi` reported CUDA 13.1
- PyTorch CUDA build: 13.0
- PyTorch reported CUDA available: True

### bitsandbytes

Standalone validation reported:

```
bitsandbytes v0.50.2
CUDA 13.0
Highest Compute Capability: (8,9)
SUCCESS!
```

This established that the RTX 4050 could use the installed bitsandbytes CUDA backend.

---

## Phase 2 — Model Loading

Model:

```
Qwen/Qwen3.5-4B-Base
```

4-bit configuration:

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
```

After resolving a transient Windows native crash and rebooting, all 426 model weight files loaded successfully.

Approximate VRAM allocation during standalone loading:

```
2.906 GB
```

---

## Phase 3 — Baseline

The first baseline generation used thinking-enabled behavior and produced reasoning-heavy output.

That configuration was rejected for the intended coding comparison.

The baseline was rerun with:

```
enable_thinking=False
```

Final baseline:

- 28 prompt tokens
- 150 generated tokens
- 15.55 seconds
- 9.65 tokens/s
- 3.152 GB peak allocated VRAM

This became the reference configuration for later evaluation.

---

## Phase 4 — Dataset Acquisition

Source:

```
sahil2801/CodeAlpaca-20k
```

Raw size:

```
20,022 records
```

The Hugging Face dataset service returned a 429 during one retrieval attempt, so the dataset's JSON file was downloaded directly from the repository.

---

## Phase 5 — Dataset Quality Audit

Results:

```
Total records: 20,022
Empty instructions: 0
Empty inputs: 9,764
Empty outputs: 6
Duplicate records: 0

Instruction:
  Min: 14
  Median: 71
  Mean: 73.8
  Max: 288

Input:
  Min: 0
  Median: 5
  Mean: 23.4
  Max: 635

Output:
  Min: 0
  Median: 131
  Mean: 197.0
  Max: 3905

Outputs under 20 chars: 1,069
```

Six empty-output records were removed.

Short outputs were retained because length alone is not a reliable quality criterion.

---

## Phase 6 — Domain Curation

Initial language heuristics produced approximately:

- Python: 7,967
- SQL: 1,892
- JavaScript: 1,670
- Java: 746
- C/C++: 462
- ML/Data: 421

A finance keyword filter initially produced a misleadingly large candidate set.

The filter was rejected because common programming words generated many false positives.

A stricter score-based system produced:

```
Python >= 3: 5731
ML >= 4: 338
ML >= 7: 175
Finance >= 4: 25
Finance >= 6: 1
Finance >= 8: 1
```

Decision:

**Python + ML/Data Science**

rather than claiming a Finance/FinTech specialization unsupported by the dataset.

---

## Phase 7 — Final Dataset

After cleaning:

```
Scored records: 20,016
Unique prompt/input pairs: 20,016
General Python examples: 5,000
ML/Data examples: 338
Total: 5,338
```

Split:

```
Train: 4,270
Validation: 534
Test: 534
```

Leakage verification:

```
Train/Validation overlap: 0
Train/Test overlap: 0
Validation/Test overlap: 0
```

The final examples were converted to conversational `messages` JSONL.

---

## Phase 8 — Token Audit Correction

The first token audit implementation incorrectly measured the length of the structured tokenizer return object.

It returned repeated values of `2`.

The issue was diagnosed and corrected by measuring:

```
tokenizer(
    tokenizer.apply_chat_template(..., tokenize=False),
    add_special_tokens=False
)["input_ids"]
```

The actual longest training example was approximately:

```
1,786 tokens
```

This result informed later memory testing.

---

## Phase 9 — Architecture Inspection

The model contained:

- 249 linear modules
- conventional attention projections
- MLP projections
- hybrid/linear-attention projections

Examples included:

```
q_proj
k_proj
v_proj
o_proj

gate_proj
up_proj
down_proj

in_proj_qkv
in_proj_z
in_proj_b
in_proj_a
out_proj
```

This made narrow attention-only targeting less representative of the architecture.

---

## Phase 10 — LoRA Verification

Configuration:

```python
r=16
lora_alpha=32
lora_dropout=0.05
bias="none"
target_modules="all-linear"
task_type="CAUSAL_LM"
```

Result:

```
Targeted modules: 248
Trainable parameters: 32,464,896
Total parameters: 4,238,216,192
Trainable fraction: 0.766%
```

---

## Phase 11 — QLoRA Dry Run

One real training example was used.

Validated:

- Forward pass
- Backward pass
- Gradients
- Optimizer step

Peak allocated VRAM:

```
4.799 GB
```

The pipeline passed the dry run.

---

## Phase 12 — Memory Stress Test

A 512-token configuration was tested.

Results:

```
Peak allocated VRAM: 6.279 GB
Peak reserved VRAM: 6.604 GB
```

This exceeded the available 6 GB GPU memory.

Decision:

```
MAX_LENGTH = 256
```

The final training run at 256 tokens later peaked at:

```
Allocated: 5.443 GB
Reserved: 5.756 GB
```

---

## Phase 13 — Training Failures

### Native access violation

One model-load run terminated with:

```
-1073741819
0xC0000005
```

A reboot resolved the transient loading issue.

### Interrupted run

An early training process stopped around step 8 before a valid checkpoint existed.

### Incomplete checkpoint

A later crash created `checkpoint-25`, but `trainer_state.json` was missing.

The checkpoint was therefore treated as invalid for automatic resume.

### Mixed-precision failure

Using trainer-level FP16 caused:

```
NotImplementedError:
"_amp_foreach_non_finite_check_and_unscale_cuda"
not implemented for 'BFloat16'
```

The final trainer configuration disabled both:

```
fp16=False
bf16=False
```

The 4-bit model still used FP16 as its compute dtype.

---

## Phase 14 — Final Training Configuration

```text
max_length = 256
per_device_train_batch_size = 1
per_device_eval_batch_size = 1
gradient_accumulation_steps = 8
num_train_epochs = 1
learning_rate = 1e-4
weight_decay = 0.01
warmup_steps = 16
lr_scheduler_type = cosine
optim = paged_adamw_8bit
gradient_checkpointing = True
use_reentrant = False
packing = False
assistant_only_loss = True
seed = 42
data_seed = 42
save_total_limit = 2
dataloader_num_workers = 0
report_to = none
push_to_hub = False
```

---

## Phase 15 — Successful Run

Final progress:

```
534 / 534
Epoch 1.0
```

Training:

```
Runtime: 5318.7316 seconds
Samples/sec: 0.803
Steps/sec: 0.100
Training loss: 0.0835487992
```

Validation near completion:

```
Eval loss: ~0.3977
Mean token accuracy: ~0.8814
Eval runtime: ~268.1 seconds
Eval samples/sec: ~1.992
Eval steps/sec: ~1.992
```

GPU:

```
Peak allocated VRAM: 5.443 GB
Peak reserved VRAM: 5.756 GB
```

Final adapter:

```
model/qwen3.5-4b-python-qlora
```

Metrics:

```
model/qwen3.5-4b-python-qlora/training_metrics.json
```

---

## Phase 16 — What the Result Does and Does Not Prove

The run proves that:

- Qwen3.5-4B-Base can be loaded in 4-bit NF4 on the RTX 4050.
- LoRA can be attached to the selected linear modules.
- The training pipeline can execute forward/backward/optimizer steps.
- A 5,338-example Python + ML/Data dataset can be prepared without detected prompt/input overlap.
- One epoch of QLoRA SFT can complete within the 6 GB hardware constraint.
- The final adapter can be saved.

The run does **not** yet prove that:

- The fine-tuned model is better than the base model.
- The model writes more executable code.
- The model generalizes to unseen programming problems.
- The model improves on HumanEval/MBPP.
- The model is better for real-world software engineering.

Those claims require the evaluation stage.

---

## Phase 17 — Evaluation Protocol

The final evaluation should use the same:

- prompts
- tokenizer
- chat template
- generation settings
- maximum generation length
- deterministic decoding configuration

for both:

```
Base model
vs.
Base + trained LoRA adapter
```

The test set contains 534 examples and was not used during training.

Evaluation outputs should be stored as structured JSON/CSV so the results can be inspected rather than relying on screenshots alone.

---

## Report Figure Mapping

The project development screenshots were captured at the following major points:

| Figure | Evidence |
|---:|---|
| 1 | PyTorch + CUDA baseline |
| 2 | NumPy installation |
| 3 | Package stack |
| 4 | Stack verification |
| 5 | bitsandbytes/NF4 validation |
| 6 | Initial Qwen 4-bit load failure |
| 7 | PyTorch / torchvision / Pillow check |
| 8 | Post-upgrade bitsandbytes test |
| 9 | Successful Qwen 4-bit load |
| 10 | Tokenizer and chat-template verification |
| 11 | Baseline V2 failure |
| 12 | Baseline V3 success |
| 13 | Dataset directory |
| 14 | Dataset download and inspection |
| 15 | Data quality audit |
| 16 | Dataset curation |
| 17 | Domain scoring |
| 18 | Final dataset construction and leakage verification |
| 19 | Final JSONL format verification |
| 20 | Architecture inspection |
| 21 | LoRA configuration and trainable parameter verification |
| 22 | QLoRA dry run |
| 23 | 512-token memory stress test |
| 24 | Training progress |
| 25 | Final training completion and metrics |

The latest final-training screenshot is the primary evidence for the completed fine-tuning stage.

Raw screenshots are not committed automatically because the GitHub connector does not have access to the conversation's image binaries as repository files. Mermaid diagrams are used directly in Markdown for portable visual documentation.

---

## Next experiment

The next repository update should contain:

1. Base-model test results.
2. Fine-tuned-model test results.
3. Per-example generation outputs.
4. Execution pass/fail results.
5. Aggregate metrics.
6. HumanEval/MBPP results where compatible.
7. Error categories.
8. A final conclusion based on measured evidence.

