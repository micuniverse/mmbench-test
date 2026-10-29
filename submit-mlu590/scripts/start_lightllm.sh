#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: start_lightllm.sh gemma|qwen36|qwen3vl" >&2
  exit 2
fi
model="$1"
submit_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fa3_root="${MMBENCH_FA3_ROOT:-$(cd "${submit_root}/.." && pwd)}"
project_root="${MMBENCH_PROJECT_ROOT:-$(cd "${fa3_root}/.." && pwd)}"
models_root="${MMBENCH_MODELS_ROOT:-/data/wangpengyu_vlm/models}"
lightllm="${MMBENCH_LIGHTLLM_DIR:-${project_root}/LightLLM-qwen3vl}"
python_bin="${MMBENCH_PYTHON:-/torch/venv3/pytorch_infer/bin/python}"

if [[ ! -f "${lightllm}/lightllm/server/api_server.py" ]]; then
  echo "LightLLM source is missing: ${lightllm}; set MMBENCH_LIGHTLLM_DIR." >&2
  exit 2
fi

export NEUWARE_HOME=/usr/local/neuware
export PATH="${NEUWARE_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${NEUWARE_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${lightllm}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export LOADWORKER=16
export LIGHTLLM_TRITON_AUTOTUNE_LEVEL=1
export LIGHTLLM_LOG_LEVEL=debug

require_model_dir() {
  if [[ ! -d "$1" ]]; then
    echo "Model directory is missing: $1" >&2
    exit 2
  fi
}

cd "${lightllm}"

# FA3 language prefill and SDPA vision match this submission's saved run.
# Qwen3-VL uses the tested Triton decode configuration.
case "${model}" in
  gemma)
    export MLU_VISIBLE_DEVICES=0
    require_model_dir "${models_root}/gemma-3-4b-it"
    exec "${python_bin}" -m lightllm.server.api_server \
      --model_dir "${models_root}/gemma-3-4b-it" \
      --tp 1 \
      --hardware_platform mlu \
      --data_type bfloat16 \
      --llm_prefill_att_backend fa3 \
      --llm_decode_att_backend fa3 \
      --vit_att_backend sdpa \
      --enable_triton_fallback \
      --mem_fraction 0.8 \
      --max_req_total_len 20480 \
      --batch_max_tokens 4096 \
      --chunked_prefill_size 2048 \
      --disable_dynamic_prompt_cache \
      --graph_max_batch_size 8 \
      --graph_max_len_in_batch 20480 \
      --trust_remote_code \
      --host 127.0.0.1 \
      --port 8088
    ;;

  qwen36)
    export MLU_VISIBLE_DEVICES=0,1
    require_model_dir "${models_root}/Qwen3.6-27B"
    exec "${python_bin}" -m lightllm.server.api_server \
      --model_dir "${models_root}/Qwen3.6-27B" \
      --tp 2 \
      --hardware_platform mlu \
      --data_type bfloat16 \
      --llm_prefill_att_backend fa3 \
      --llm_decode_att_backend fa3 \
      --vit_att_backend sdpa \
      --enable_triton_fallback \
      --mem_fraction 0.8 \
      --max_req_total_len 20480 \
      --batch_max_tokens 4096 \
      --chunked_prefill_size 2048 \
      --running_max_req_size 16 \
      --graph_max_batch_size 16 \
      --graph_max_len_in_batch 20480 \
      --trust_remote_code \
      --reasoning_parser qwen3 \
      --host 127.0.0.1 \
      --port 8089
    ;;

  qwen3vl)
    export MLU_VISIBLE_DEVICES=0
    require_model_dir "${models_root}/Qwen3-VL-8B-Instruct"
    exec "${python_bin}" -m lightllm.server.api_server \
      --model_dir "${models_root}/Qwen3-VL-8B-Instruct" \
      --tp 1 \
      --hardware_platform mlu \
      --data_type bfloat16 \
      --llm_prefill_att_backend fa3 \
      --llm_decode_att_backend triton \
      --vit_att_backend sdpa \
      --enable_triton_fallback \
      --mem_fraction 0.8 \
      --max_req_total_len 20480 \
      --batch_max_tokens 4096 \
      --chunked_prefill_size 2048 \
      --running_max_req_size 16 \
      --graph_max_batch_size 16 \
      --graph_max_len_in_batch 20480 \
      --trust_remote_code \
      --disable_cudagraph \
      --host 127.0.0.1 \
      --port 8090
    ;;

  *)
    echo "unknown model: ${model}" >&2
    exit 2
    ;;
esac
