#!/usr/bin/env bash
# Zero-shot evaluation of the original Qwen2.5-3B-Instruct checkpoint (Base).
set -euo pipefail
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
