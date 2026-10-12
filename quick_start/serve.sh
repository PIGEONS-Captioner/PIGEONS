#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "--help" || $# -eq 0 ]]; then
  echo 'Usage: bash quick_start/serve.sh /path/to/PIGEONS-checkpoint [extra swift deploy arguments]'
  echo 'Environment: PORT=12346 TP=2 MAX_MODEL_LEN=40960 CUDA_VISIBLE_DEVICES=0,1'
  echo 'Set generation options (temperature, top_k, top_p, max-tokens, seed) on quick_start/inference.py.'
  exit 0
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="$1"
shift
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Model directory does not exist: $MODEL_PATH" >&2
  exit 2
fi
export ENABLE_AUDIO_OUTPUT=0 IMAGE_FACTOR=32 MAX_PIXELS=262144 VIDEO_MAX_PIXELS=262144
export VIDEO_TOTAL_PIXELS=31457280 USE_AUDIO_IN_VIDEO=True FPS=2
# Generation settings come from each client request; seed also initializes the server.
exec swift deploy --model "$MODEL_PATH" --served_model_name PIGEONS \
  --external_plugins "$SCRIPT_DIR/../pigeons/swift_plugin.py" \
  --infer_backend vllm --host 127.0.0.1 --port "${PORT:-12346}" \
  --seed 42 \
  --vllm_max_model_len "${MAX_MODEL_LEN:-40960}" \
  --vllm_tensor_parallel_size "${TP:-2}" --vllm_data_parallel_size 1 \
  --vllm_gpu_memory_utilization 0.85 --vllm_max_num_seqs 1 \
  "$@"
