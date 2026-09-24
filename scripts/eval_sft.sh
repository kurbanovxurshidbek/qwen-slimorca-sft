#!/usr/bin/env bash
# Zero-shot evaluation of the fine-tuned checkpoint (SFT) saved by Axolotl in /workspace/outputs.
set -euo pipefail
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
