#!/usr/bin/env bash
set -euo pipefail

submit_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fa3_root="${MMBENCH_FA3_ROOT:-$(cd "${submit_root}/.." && pwd)}"
project_root="${MMBENCH_PROJECT_ROOT:-$(cd "${fa3_root}/.." && pwd)}"
python_bin="${MMBENCH_PYTHON:-/torch/venv3/pytorch_infer/bin/python}"
export MMBENCH_FA3_ROOT="${fa3_root}"
export MMBENCH_PROJECT_ROOT="${project_root}"
for required in curl setsid realpath; do
  if ! command -v "${required}" >/dev/null; then
    echo "Required command is missing: ${required}" >&2
    exit 2
  fi
done
run_root="$(realpath -m "${1:-${fa3_root}/runs/fa3_recheck_$(date +%Y%m%d_%H%M%S)}")"
if [[ ! -x "${python_bin}" ]]; then
  echo "Python interpreter is missing or not executable: ${python_bin}" >&2
  exit 2
fi
if [[ -e "${run_root}" ]]; then
  echo "Use a fresh run directory: ${run_root}" >&2
  exit 2
fi
if [[ ! -f "${fa3_root}/dataset/prepared_messages_200_letter_v1.jsonl" ]] ||
   [[ ! -f "${fa3_root}/dataset/prepared_messages_200_letter_v1.meta.json" ]]; then
  echo "Prepared VLMEvalKit messages are missing from ${fa3_root}/dataset." >&2
  exit 2
fi
if [[ ! -f "${MMBENCH_LIGHTLLM_DIR:-${project_root}/LightLLM-qwen3vl}/lightllm/server/api_server.py" ]]; then
  echo "LightLLM source is missing; set MMBENCH_LIGHTLLM_DIR." >&2
  exit 2
fi

export NEUWARE_HOME="${NEUWARE_HOME:-/usr/local/neuware}"
export LD_LIBRARY_PATH="${NEUWARE_HOME}/lib64:${LD_LIBRARY_PATH:-}"
mkdir -p "${run_root}"
server_pid=""

stop_server() {
  if [[ -n "${server_pid}" ]] && kill -0 -- "-${server_pid}" 2>/dev/null; then
    kill -INT -- "-${server_pid}" 2>/dev/null || true
    for _ in {1..15}; do
      if ! kill -0 -- "-${server_pid}" 2>/dev/null; then break; fi
      sleep 1
    done
    if kill -0 -- "-${server_pid}" 2>/dev/null; then
      kill -TERM -- "-${server_pid}" 2>/dev/null || true
      for _ in {1..5}; do
        if ! kill -0 -- "-${server_pid}" 2>/dev/null; then break; fi
        sleep 1
      done
    fi
    if kill -0 -- "-${server_pid}" 2>/dev/null; then
      kill -KILL -- "-${server_pid}" 2>/dev/null || true
    fi
    wait "${server_pid}" 2>/dev/null || true
  fi
  server_pid=""
}
trap stop_server EXIT

run_model() {
  local name="$1" port="$2" model_id="$3"
  local model_dir="${run_root}/${name}"
  local url="http://127.0.0.1:${port}/v1/chat/completions"
  local effort=()
  if [[ "${name}" == qwen36 ]]; then effort=(--reasoning-effort none); fi
  if curl -fsS --max-time 2 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
    echo "Port ${port} already has a healthy service; refusing to mix runs." >&2
    return 2
  fi
  mkdir -p "${model_dir}"
  echo "Starting ${name} on port ${port}; log: ${model_dir}/server.log"
  setsid bash "${submit_root}/scripts/start_lightllm.sh" "${name}" > "${model_dir}/server.log" 2>&1 &
  server_pid=$!
  local ready=0
  for _ in {1..450}; do
    if curl -fsS --max-time 2 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      ready=1
      break
    fi
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      echo "${name} service exited; inspect ${model_dir}/server.log" >&2
      return 1
    fi
    sleep 2
  done
  if [[ "${ready}" -ne 1 ]]; then
    echo "${name} service did not become healthy in 15 minutes." >&2
    return 1
  fi
  echo "${name} is healthy; running preflight and full validation"
  if [[ "${name}" == qwen36 ]]; then
    "${python_bin}" "${submit_root}/scripts/mmbench_openai_runner.py" probe-reasoning \
      --url "${url}" --model "${model_id}" \
      --output "${model_dir}/reasoning_probe.json"
  fi
  "${python_bin}" "${submit_root}/scripts/mmbench_openai_runner.py" run \
    --url "${url}" --model "${model_id}" "${effort[@]}" \
    --limit 2 --output "${model_dir}/two"
  "${python_bin}" "${submit_root}/scripts/mmbench_openai_runner.py" run \
    --url "${url}" --model "${model_id}" "${effort[@]}" \
    --limit 200 --output "${model_dir}/full"
  echo "${name} completed; stopping its service"
  stop_server
}

run_model gemma 8088 gemma-3-4b-it
run_model qwen36 8089 Qwen3.6-27B
run_model qwen3vl 8090 Qwen3-VL-8B-Instruct
echo "All new runs are in ${run_root}"
