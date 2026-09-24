# Task-Dependent Effects of SlimOrca SFT on Qwen2.5-3B-Instruct

Experiment artifacts for the paper:

> K. Kurbanov, *Task-Dependent Effects of SlimOrca Supervised Fine-Tuning on Qwen2.5-3B-Instruct*, 2026 (under review).

The study fine-tunes **Qwen/Qwen2.5-3B-Instruct** on a subset of **Open-Orca/SlimOrca** and compares the original (Base) and fine-tuned (SFT) checkpoints on six benchmarks under an identical zero-shot evaluation protocol.

## Repository contents

| Path | Description |
|---|---|
| `prepare_dataset.ipynb` | Loads SlimOrca, draws a random subset (`shuffle(seed=42)`, 5,000 examples) and converts conversations to chat `messages` format |
| `train.jsonl` | The 5,000 training examples produced by the notebook |
| `qwen_slimorca_sft.yaml` | Axolotl training configuration |
| `eval_results/base/` | lm-evaluation-harness outputs for the Base model (aggregate results and per-sample logs) |
| `eval_results/sft/` | lm-evaluation-harness outputs for the SFT model |
| `scripts/eval_base.sh`, `scripts/eval_sft.sh` | lm-evaluation-harness commands used for the Base and SFT evaluations |
| `scripts/summarize_results.py` | Builds the Base-vs-SFT comparison table from `eval_results/` |
| `results/summary.csv`, `results/summary.md` | Output of the summary script |

## Training setup

Taken from `qwen_slimorca_sft.yaml`:

| Parameter | Value |
|---|---|
| Base model | Qwen/Qwen2.5-3B-Instruct |
| Fine-tuning method | Full-parameter SFT (no LoRA/QLoRA, no 4/8-bit loading) |
| Training data | `train.jsonl` (5,000 SlimOrca examples), `chat_template` format |
| Validation split | 1% (`val_set_size: 0.01`) |
| Maximum sequence length | 8,192 tokens, no sample packing |
| Micro-batch size / gradient accumulation | 8 / 4 (effective batch size 32 on one GPU) |
| Epochs | 3 |
| Optimizer / scheduler | AdamW (fused) / cosine |
| Learning rate / warm-up / weight decay | 2e-5 / 5% / 0.01 |
| Precision | BF16 (TF32 enabled) |
| Attention | FlashAttention-2 |
| Gradient checkpointing | Enabled |
| Hardware | 1× NVIDIA H200 (RunPod) |

Training:

```bash
axolotl train qwen_slimorca_sft.yaml
```

Training was logged to Weights & Biases (project `qwen2.5-sft`, run `qwen2.5-3b-slimorca-5k`). The configuration pushes the fine-tuned model to the Hugging Face Hub as `kurbanovxurshidbek/qwen2.5-3b-slimorca-sft`.

## Evaluation setup

Both checkpoints were evaluated with [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) **0.4.11** (Transformers 5.16.1) using the Hugging Face backend, bfloat16, zero-shot, the Qwen chat template, automatic batch size and a single CUDA device. GSM8K uses greedy decoding (`temperature=0`) with stop sequences `Question:`, `</s>`, `<|im_end|>`.

**1. Install lm-evaluation-harness**

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

**2. Evaluate the Base model** ([`scripts/eval_base.sh`](scripts/eval_base.sh))

```bash
mkdir -p /workspace/eval_results/base

lm-eval run \
  --model hf \
  --model_args pretrained=Qwen/Qwen2.5-3B-Instruct,dtype=bfloat16 \
  --tasks arc_easy,arc_challenge,hellaswag,mmlu,winogrande,gsm8k \
  --num_fewshot 0 \
  --device cuda:0 \
  --batch_size auto \
  --apply_chat_template \
  --output_path /workspace/eval_results/base \
  --log_samples
```

**3. Evaluate the fine-tuned model** ([`scripts/eval_sft.sh`](scripts/eval_sft.sh)); `/workspace/outputs` is the Axolotl output directory

```bash
mkdir -p /workspace/eval_results/sft

lm-eval run \
  --model hf \
  --model_args pretrained=/workspace/outputs,dtype=bfloat16 \
  --tasks arc_easy,arc_challenge,hellaswag,mmlu,winogrande,gsm8k \
  --num_fewshot 0 \
  --device cuda:0 \
  --batch_size auto \
  --apply_chat_template \
  --output_path /workspace/eval_results/sft \
  --log_samples
```

## Results

Reproduce the table with `python scripts/summarize_results.py`. Scores are in percent with the standard errors reported by the harness; Δ is the difference in percentage points.

| Benchmark | Metric | Base (%) | SFT (%) | Δ (pp) |
|---|---|---|---|---|
| ARC-Easy | acc_norm | 54.59 ± 1.02 | 74.20 ± 0.90 | +19.61 |
| ARC-Challenge | acc_norm | 44.20 ± 1.45 | 49.83 ± 1.46 | +5.63 |
| HellaSwag | acc_norm | 64.34 ± 0.48 | 68.96 ± 0.46 | +4.62 |
| MMLU | acc | 64.48 ± 0.38 | 65.10 ± 0.38 | +0.62 |
| WinoGrande | acc | 63.14 ± 1.36 | 65.82 ± 1.33 | +2.68 |
| GSM8K | exact_match (flexible-extract) | 45.87 ± 1.37 | 55.04 ± 1.37 | +9.17 |

MMLU by category:

| Category | Base (%) | SFT (%) | Δ (pp) |
|---|---|---|---|
| Humanities | 57.64 | 57.90 | +0.26 |
| Other | 70.33 | 70.49 | +0.16 |
| Social Sciences | 75.14 | 75.79 | +0.65 |
| STEM | 58.52 | 60.13 | +1.61 |

## Notes

- `eval_results/sft/__workspace__outputs/` is named after the local checkpoint path (`/workspace/outputs`) used on RunPod.
- The per-sample `samples_*.jsonl` files contain every prompt, model response and score, allowing item-level re-analysis.
- The trained model weights are not stored in this repository.

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

Code and configuration files are released under the MIT License (see [`LICENSE`](LICENSE)). The training data is derived from [Open-Orca/SlimOrca](https://huggingface.co/datasets/Open-Orca/SlimOrca) and remains subject to that dataset's license.
