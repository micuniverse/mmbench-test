#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
work_dir="$(cd "$script_dir/../.." && pwd)"
project_dir="$(cd "$work_dir/.." && pwd)"
results_dir="${MMBENCH_RESULTS_DIR:-$work_dir/submit/results}"
logs_dir="$work_dir/logs"
selection="${1:-all}"
case "$selection" in
  all|gemma3|qwen3.6|qwen3vl) ;;
  -h|--help) echo "Usage: $0 [all|gemma3|qwen3.6|qwen3vl]"; exit 0 ;;
  *) echo "Usage: $0 [all|gemma3|qwen3.6|qwen3vl]" >&2; exit 2 ;;
esac
if ! python_bin="$(command -v "${PYTHON:-/opt/conda/bin/python3}")"; then
  echo "Python not found" >&2
  exit 2
fi
[[ -x "$script_dir/start_lightllm.sh" ]] || { echo "start_lightllm.sh is missing" >&2; exit 2; }
[[ -f "$script_dir/run_framework_eval.py" ]] || { echo "run_framework_eval.py is missing" >&2; exit 2; }
mkdir -p "$results_dir" "$logs_dir"
export PATH="$(dirname "$python_bin"):$project_dir/scripts/cci_bin:$project_dir/.cci_deps/bin:$PATH"
export PYTHONPATH="$project_dir/.cci_deps${PYTHONPATH:+:$PYTHONPATH}"

pid=""
group_ready=0
group_alive() {
  ps -eo pgid=,stat= | awk -v id="$pid" \
    '$1 == id && $2 !~ /^Z/ { live = 1 } END { exit !live }'
}

stop_service() {
  [[ -n "$pid" ]] || return 0
  if (( group_ready )); then
    kill -TERM -- "-$pid" 2>/dev/null || true
    for ((i = 0; i < 15; i++)); do
      group_alive || break
      sleep 1
    done
    if group_alive; then
      kill -KILL -- "-$pid" 2>/dev/null || true
      for ((i = 0; i < 5; i++)); do
        group_alive || break
        sleep 1
      done
    fi
  else
    kill -TERM "$pid" 2>/dev/null || true
  fi
  wait "$pid" 2>/dev/null || true
  if (( group_ready )) && group_alive; then
    echo "LightLLM workers are still running in process group $pid" >&2
    return 1
  fi
  pid=""
  group_ready=0
}
trap 'stop_service || true' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_for_model() {
  local model_id="$1" port="$2" log="$3"
  for ((i = 0; i < 450; i++)); do
    if curl --noproxy '*' -fsS --max-time 3 "http://127.0.0.1:$port/v1/models" 2>/dev/null |
      "$python_bin" -c '
import json, sys
models = json.load(sys.stdin).get("data", [])
found = any(isinstance(item, dict) and item.get("id") == sys.argv[1] for item in models)
sys.exit(0 if found else 1)
' "$model_id" 2>/dev/null; then
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "LightLLM exited before becoming ready; see $log" >&2
      return 1
    fi
    sleep 2
  done
  echo "Timed out waiting for $model_id; see $log" >&2
  return 1
}

run_one() {
  local model="$1" model_id port devices extra_body
  case "$model" in
    gemma3)
      model_id=Gemma-3-4B-IT
      port=8000
      devices=0
      extra_body='{}'
      ;;
    qwen3.6)
      model_id=Qwen3.6-27B
      port=8001
      devices=1,2
      extra_body='{"reasoning_effort":"none"}'
      ;;
    qwen3vl)
      model_id=Qwen3-VL-8B-Instruct
      port=8002
      devices=0,1
      extra_body='{}'
      ;;
  esac

  local result_dir="$results_dir/$model_id"
  [[ ! -e "$result_dir" ]] || { echo "Result already exists: $result_dir" >&2; return 1; }
  if (echo >/dev/tcp/127.0.0.1/"$port") 2>/dev/null; then
    echo "Port $port is already in use" >&2
    return 1
  fi

  local stamp service_log eval_log pgid
  stamp="$(date -u +%Y%m%dT%H%M%S%NZ)"
  service_log="$logs_dir/${model//./_}_$stamp.service.log"
  eval_log="$logs_dir/${model//./_}_$stamp.eval.log"
  echo "Starting $model_id on GPU $devices"

  # The wrapper keeps the service PID equal to its process-group ID.
  MMBENCH_HOST=127.0.0.1 MMBENCH_PORT="$port" \
    CUDA_VISIBLE_DEVICES="$devices" PYTHON="$python_bin" \
    "$python_bin" -c \
      'import os,sys; os.setsid(); os.execv(sys.argv[1],sys.argv[1:])' \
      "$script_dir/start_lightllm.sh" "$model" >"$service_log" 2>&1 &
  pid=$!
  for ((i = 0; i < 50; i++)); do
    pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d '[:space:]' || true)"
    if [[ "$pgid" == "$pid" ]]; then group_ready=1; break; fi
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.1
  done
  (( group_ready )) || { echo "Service launch failed; see $service_log" >&2; return 1; }

  wait_for_model "$model_id" "$port" "$service_log"
  grep -Fq "Auto-selected flashinfer backend" "$service_log" ||
    { echo "FlashInfer was not selected; see $service_log" >&2; return 1; }
  if [[ "$model" == qwen3vl ]]; then
    grep -Fq "Auto-selected xformers backend" "$service_log" ||
      { echo "xformers VIT backend was not selected; see $service_log" >&2; return 1; }
  fi

  "$python_bin" -u "$script_dir/run_framework_eval.py" \
    --base-url "http://127.0.0.1:$port/v1" \
    --model "$model_id" \
    --limit 200 \
    --workers 4 \
    --max-tokens 1024 \
    --extra-body "$extra_body" \
    --output-dir "$result_dir" 2>&1 | tee "$eval_log"

  stop_service
  # Keep raw model responses and the per-question score table only.
  find "$result_dir" -maxdepth 1 -type f ! -name 'predictions_exact_matching_result.tsv' -delete
  find "$result_dir/cases" -maxdepth 1 -type f ! -name '*_response.json' -delete
  echo "Result: $result_dir"
}

if [[ "$selection" == all ]]; then
  run_one gemma3
  run_one qwen3.6
  run_one qwen3vl
else
  run_one "$selection"
fi
