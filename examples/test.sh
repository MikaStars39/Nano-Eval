#!/usr/bin/env bash
set -euo pipefail

# Real offline evaluation for math tasks with a local model.
REPO_ROOT=/mnt/llm-train/users/explore-train/qingyu/NanoEval
WORKDIR=${WORKDIR:-${REPO_ROOT}/outputs/test_offline_aime2024}
PREPARED_INPUT="${WORKDIR}/step01_prepared.jsonl"
INFERENCE_OUTPUT="${WORKDIR}/step02_inference.jsonl"
SCORE_OUTPUT="${WORKDIR}/step03_score.jsonl"
FINAL_EVAL_OUTPUT="${WORKDIR}/step03_final_eval.jsonl"
LOG_FILE="${WORKDIR}/run.log"
mkdir -p "${WORKDIR}"

TASK_ARGS=(
  --stage all
  --task-dir ${REPO_ROOT}/outputs/nano_eval
  --tasks "aime2024@4,aime2025@4,math500@1,gpqa_diamond@1"
  --pass-k 1
  --output ${PREPARED_INPUT}
  --inference-output ${INFERENCE_OUTPUT}
  --score-output ${SCORE_OUTPUT}
  --final-eval-output ${FINAL_EVAL_OUTPUT}
  --system-prompt ""
  --n-proc 8
)

ROLLOUT_ARGS=(
  --backend offline
  --model-path /mnt/llm-train/users/explore-train/qingyu/.cache/DeepSeek-R1-Distill-Qwen-1.5B
  --work-dir ${WORKDIR}
  # Added: dedicated model for chat template in step01; defaults to --model-path.
  # --chat-template-model-path /path/to/chat-template-model
  # Added: API key for online backend only.
  # --api-key "YOUR_API_KEY"
  # Added: API base URL for online backend only.
  # --base-url "https://api.example.com/v1"
  # Added: remote model name for online backend only.
  # --model "gpt-4o-mini"
  --temperature 1
  --max-tokens 32768
  --concurrency 32
)

python "${REPO_ROOT}/run.py" \
  "${TASK_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
