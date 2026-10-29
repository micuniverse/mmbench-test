#!/usr/bin/env bash
set -euo pipefail

# A100 - 40g.docx launch settings, with vision enabled for MMBench.
# Qwen3-VL is configured here because that model is absent from the document.

model=${1:-}
case "$model" in
  gemma3|qwen3.6|qwen3vl) ;;
  -h|--help) echo "Usage: $0 gemma3|qwen3.6|qwen3vl"; exit 0 ;;
  *) echo "Usage: $0 gemma3|qwen3.6|qwen3vl" >&2; exit 2 ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "$script_dir/../../.." && pwd)"
lightllm_dir="${MMBENCH_LIGHTLLM_ROOT:-$project_dir/frameworks/LightLLM}"
models_dir="${MMBENCH_MODELS_ROOT:-/data/models}"
if ! python_bin="$(command -v "${PYTHON:-/opt/conda/bin/python3}")"; then
  echo "Python not found" >&2
  exit 2
fi

case "$model" in
  gemma3)
    model_dir="$models_dir/Gemma-3-4B-IT"
    default_port=8000
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    ;;
  qwen3.6)
    model_dir="$models_dir/Qwen3.6-27B"
    default_port=8001
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"
    ;;
  qwen3vl)
    model_dir="$models_dir/Qwen3-VL-8B-Instruct"
    default_port=8002
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
    ;;
esac
host="${MMBENCH_HOST:-127.0.0.1}"
port="${MMBENCH_PORT:-$default_port}"

[[ -f "$lightllm_dir/lightllm/server/api_server.py" ]] || { echo "LightLLM source missing: $lightllm_dir" >&2; exit 2; }
[[ -d "$model_dir" ]] || { echo "Model missing: $model_dir" >&2; exit 2; }

export PATH="$(dirname "$python_bin"):$project_dir/scripts/cci_bin:$project_dir/.cci_deps/bin:$PATH"
export PYTHONPATH="$lightllm_dir:$project_dir/.cci_deps${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$project_dir/.cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$XDG_CACHE_HOME/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$XDG_CACHE_HOME/torchinductor}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$XDG_CACHE_HOME/flashinfer}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE"

"$python_bin" - <<'PY'
import sys
import torch
if sys.version_info[:2] not in ((3, 10), (3, 11)):
    raise SystemExit(f"Expected Python 3.10/3.11, got {sys.version.split()[0]}")
if not torch.__version__.startswith("2.11.0") or torch.version.cuda != "13.0":
    raise SystemExit(f"Expected torch 2.11.0/CUDA 13.0, got {torch.__version__}/{torch.version.cuda}")
PY

cd "$lightllm_dir"
case "$model" in
  gemma3)
    # MMBench needs vision, so do not use the document's --disable_vision.
    exec "$python_bin" -m lightllm.server.api_server \
      --model_dir "$model_dir" \
      --tp 1 \
      --trust_remote_code \
      --max_req_total_len 20480 \
      --mem_fraction 0.72 \
      --running_max_req_size 16 \
      --disable_cudagraph \
      --disable_dynamic_prompt_cache \
      --host "$host" \
      --port "$port"
    ;;
  qwen3.6)
    exec "$python_bin" -m lightllm.server.api_server \
      --model_dir "$model_dir" \
      --tp 2 \
      --trust_remote_code \
      --max_req_total_len 20480 \
      --mem_fraction 0.8 \
      --running_max_req_size 16 \
      --graph_max_batch_size 16 \
      --linear_att_ssm_data_type bfloat16 \
      --llm_decode_att_backend flashinfer triton \
      --host "$host" \
      --port "$port"
    ;;
  qwen3vl)
    exec "$python_bin" -m lightllm.server.api_server \
      --model_dir "$model_dir" \
      --tp 2 \
      --visual_tp 1 \
      --trust_remote_code \
      --max_req_total_len 20480 \
      --mem_fraction 0.8 \
      --running_max_req_size 16 \
      --disable_cudagraph \
      --host "$host" \
      --port "$port"
    ;;
esac
