#!/usr/bin/env bash
# Run one LightLLM service in the foreground, inside the pre-created C500 container.
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
project_root="$(cd "$script_dir/../.." && pwd)"
model_root="${MMBENCH_MODEL_ROOT:-/data/models}"
model="${1:?Usage: start_lightllm.sh gemma|qwen36|qwen3vl}"

case "$model" in
  gemma)
    model_id=google/gemma-3-4b-it
    model_leaf=gemma-3-4b-it
    devices=4
    port=19301
    tp=1
    runtime_default="$project_root/runtime/LightLLM-fa3-gemma"
    model_args=(
      --mem_fraction 0.75
      --max_req_total_len 20480
      --batch_max_tokens 16384
      --chunked_prefill_size 8192
      --running_max_req_size 128
      --graph_max_batch_size 64
      --graph_max_len_in_batch 20480
      --use_tgi_api
      --disable_dynamic_prompt_cache
    )
    ;;
  qwen36)
    model_id=Qwen/Qwen3.6-27B
    model_leaf=Qwen3.6-27B
    devices=5,6
    port=19303
    tp=2
    runtime_default="$project_root/runtime/LightLLM-fa3-original"
    model_args=(
      --mem_fraction 0.80
      --max_req_total_len 262144
      --running_max_req_size 16
      --graph_max_batch_size 16
    )
    ;;
  qwen3vl)
    model_id=Qwen/Qwen3-VL-8B-Instruct
    model_leaf=Qwen3-VL-8B-Instruct
    devices=7
    port=19304
    tp=1
    runtime_default="$project_root/runtime/LightLLM-fa3-qwen3vl"
    model_args=(
      --mem_fraction 0.75
      --max_req_total_len 20480
      --batch_max_tokens 16384
      --chunked_prefill_size 8192
      --running_max_req_size 16
      --graph_max_batch_size 16
      --disable_cudagraph
      --disable_dynamic_prompt_cache
    )
    ;;
  *) echo "Unknown model key: $model" >&2; exit 2 ;;
esac

runtime="${MMBENCH_LIGHTLLM_SOURCE:-$runtime_default}"
port="${MMBENCH_PORT:-$port}"
[[ -f "$runtime/lightllm/server/api_cli.py" ]] || { echo "Missing LightLLM source: $runtime" >&2; exit 2; }
[[ -f "$model_root/$model_leaf/config.json" ]] || { echo "Missing model config: $model_root/$model_leaf" >&2; exit 2; }

export CUDA_VISIBLE_DEVICES="$devices"
export MACA_PATH=/opt/maca
export MACA_SMALL_PAGESIZE_ENABLE=1
export TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP=1
export TRITON_ENABLE_MACA_CHAIN_DOT_OPT=1
export PYTORCH_ENABLE_PG_HIGH_PRIORITY_STREAM=1
export MACA_QUEUE_SCHEDULE_POLICY=1
export MACA_GRAPH_LAUNCH_MODE=1
export MACA_DIRECT_DISPATCH=1
export MACA_TORCH_COMPILE_CONF=triton.multi_kernel:1
export LIGHTLLM_TRITON_AUTOTUNE_LEVEL=1
export LOADWORKER=16
export PAGE_SIZE=128
export HF_HOME="$project_root/runs/cache/$model/huggingface"
export TRITON_CACHE_DIR="$project_root/runs/cache/$model/triton"
mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR"
cd "$runtime"

exec /usr/bin/python -m lightllm.server.api_server \
  --model_dir "$model_root/$model_leaf" \
  --model_name "$model_id" \
  --trust_remote_code \
  --host 127.0.0.1 \
  --port "$port" \
  --tp "$tp" \
  --data_type bfloat16 \
  --quant_type none \
  "${model_args[@]}" \
  --hardware_platform maca \
  --llm_prefill_att_backend fa3 \
  --llm_decode_att_backend fa3 \
  --vit_att_backend sdpa
