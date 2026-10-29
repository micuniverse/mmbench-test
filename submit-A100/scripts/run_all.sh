#!/usr/bin/env bash
set -euo pipefail

# Each model: start LightLLM, wait for its OpenAI endpoint, run two questions,
# then run the same 200 questions and score with VLMEvalKit.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
work_dir="$(cd "$script_dir/../.." && pwd)"
project_dir="$(cd "$work_dir/.." && pwd)"
python_bin="${MMBENCH_PYTHON:-/opt/conda/bin/python3}"
runner="$script_dir/mmbench_openai_runner.py"
run_root="${MMBENCH_RUN_ROOT:-$work_dir/runs/a100_unified_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
smoke_only="${MMBENCH_SMOKE_ONLY:-0}"
selection="${1:-all}"

case "$selection" in
  all|gemma|qwen36|qwen3vl) ;;
  *) echo "Usage: $0 [all|gemma|qwen36|qwen3vl]" >&2; exit 2 ;;
esac
[[ -x "$python_bin" && -f "$runner" ]] || { echo "Python or runner is missing" >&2; exit 2; }
[[ -f "$work_dir/dataset/prepared_messages_200_letter_v1.jsonl" ]] ||
  { echo "Prepared MMBench messages are missing" >&2; exit 2; }
[[ ! -e "$run_root" ]] || { echo "Run directory exists: $run_root" >&2; exit 2; }
mkdir -p "$run_root"

export PATH="$(dirname "$python_bin"):$project_dir/scripts/cci_bin:$project_dir/.cci_deps/bin:$PATH"
export PYTHONPATH="$project_dir/.eval_deps:$project_dir/.cci_deps${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1

server_pid=""
stop_server() {
  [[ -n "$server_pid" ]] || return 0
  kill -TERM -- "-$server_pid" 2>/dev/null || true
  for ((i=0; i<15; i++)); do
    if ! ps -eo pgid=,stat= | awk -v id="$server_pid" '$1==id && $2 !~ /^Z/ {found=1} END {exit !found}'; then break; fi
    sleep 1
  done
  if ps -eo pgid=,stat= | awk -v id="$server_pid" '$1==id && $2 !~ /^Z/ {found=1} END {exit !found}'; then
    kill -KILL -- "-$server_pid" 2>/dev/null || true
  fi
  wait "$server_pid" 2>/dev/null || true
  server_pid=""
}
trap stop_server EXIT

wait_for_model() {
  local model_id="$1" port="$2" log="$3"
  for ((i=0; i<450; i++)); do
    if curl --noproxy '*' -fsS --max-time 3 "http://127.0.0.1:$port/v1/models" 2>/dev/null |
      "$python_bin" -c 'import json,sys; sys.exit(0 if any(x.get("id")==sys.argv[1] for x in json.load(sys.stdin).get("data",[])) else 1)' "$model_id" 2>/dev/null; then
      return 0
    fi
    kill -0 "$server_pid" 2>/dev/null || { echo "Service exited: $log" >&2; return 1; }
    sleep 2
  done
  echo "Service did not become ready: $log" >&2
  return 1
}

run_model() {
  local name="$1" model_id port devices service_name effort=()
  case "$name" in
    gemma)   model_id=Gemma-3-4B-IT;        port=8000; devices=0;   service_name=gemma3 ;;
    qwen36)  model_id=Qwen3.6-27B;          port=8001; devices=1,2; service_name=qwen3.6; effort=(--reasoning-effort none) ;;
    qwen3vl) model_id=Qwen3-VL-8B-Instruct; port=8002; devices=0,1; service_name=qwen3vl ;;
  esac
  local model_dir="$run_root/$name" url="http://127.0.0.1:$port/v1/chat/completions"
  mkdir -p "$model_dir"
  if (echo >/dev/tcp/127.0.0.1/"$port") 2>/dev/null; then
    echo "Port $port is occupied" >&2; return 1
  fi
  echo "Starting $model_id on GPU $devices"
  CUDA_VISIBLE_DEVICES="$devices" MMBENCH_PORT="$port" PYTHON="$python_bin" \
    "$python_bin" -c 'import os,sys; os.setsid(); os.execv(sys.argv[1],sys.argv[1:])' \
    "$script_dir/start_lightllm.sh" "$service_name" >"$model_dir/server.log" 2>&1 &
  server_pid=$!
  wait_for_model "$model_id" "$port" "$model_dir/server.log"
  grep -Fq "Auto-selected flashinfer backend" "$model_dir/server.log" ||
    { echo "FlashInfer was not selected: $model_dir/server.log" >&2; return 1; }
  if [[ "$name" == qwen36 ]]; then
    "$python_bin" "$runner" probe-reasoning --url "$url" --model "$model_id" \
      --output "$model_dir/reasoning_probe.json"
  fi
  "$python_bin" "$runner" run --url "$url" --model "$model_id" "${effort[@]}" \
    --limit 2 --output "$model_dir/two"
  if [[ "$smoke_only" != 1 ]]; then
    "$python_bin" "$runner" run --url "$url" --model "$model_id" "${effort[@]}" \
      --limit 200 --output "$model_dir/full"
  fi
  stop_server
}

if [[ "$selection" == all ]]; then
  run_model gemma
  run_model qwen36
  run_model qwen3vl
else
  run_model "$selection"
fi

# Submit the raw model responses and per-question scores. Detailed run
# artifacts remain under runs/ for diagnosis.
if [[ "$smoke_only" != 1 ]]; then
  results="$work_dir/submit/results"
  backup="$run_root/previous_submit_results"
  if [[ "$selection" == all ]]; then
    if [[ -d "$results" ]]; then mv "$results" "$backup"; fi
    mkdir -p "$results"
  fi
  for model_dir in "$run_root"/*/full; do
    [[ -d "$model_dir" ]] || continue
    name="$(basename "$(dirname "$model_dir")")"
    if [[ -d "$results/$name" ]]; then
      mkdir -p "$backup"
      mv "$results/$name" "$backup/$name"
    fi
    mkdir -p "$results/$name"
    for file in raw_requests_responses.jsonl \
                normalized_predictions_exact_matching_result.tsv; do
      cp "$model_dir/$file" "$results/$name/$file"
    done
  done
fi
echo "Run artifacts: $run_root"
