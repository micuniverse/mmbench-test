#!/usr/bin/env bash
# Run the independent 2-question precheck and 200-question MMBench test for each model.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

script_dir="$(cd "$(dirname "$0")" && pwd)"
project_root="$(cd "$script_dir/../.." && pwd)"
python_bin="${MMBENCH_PYTHON:-/data/envs/hunyuan3d-metax/bin/python}"
runner="$script_dir/mmbench_openai_runner.py"
container="${MMBENCH_CONTAINER:-c500-mmbench-fa3-submit}"
session="${1:-submit_fa3_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
[[ "$session" =~ ^[A-Za-z0-9_.-]{1,64}$ ]] || { echo "Invalid session name: $session" >&2; exit 2; }
run_root="$project_root/runs/$session"
[[ ! -e "$run_root" ]] || { echo "Run directory already exists: $run_root" >&2; exit 2; }
[[ -x "$python_bin" ]] || { echo "Missing Python: $python_bin" >&2; exit 2; }
if ! container_id="$(docker inspect --format '{{.Id}}' "$container" 2>/dev/null)"; then
  echo "Create the dedicated container first, then set MMBENCH_CONTAINER to its name" >&2
  exit 2
fi
role="$(docker inspect --format '{{ index .Config.Labels "mmbench.role" }}' "$container_id")"
[[ "$role" == fa3-submit ]] || { echo "Refusing non-FA3 test container: $container" >&2; exit 2; }
network="$(docker inspect --format '{{.HostConfig.NetworkMode}}' "$container_id")"
[[ "$network" == host ]] || { echo "Container must use host networking: $container" >&2; exit 2; }
state="$(docker inspect --format '{{.State.Status}}' "$container_id")"
[[ "$state" == created || "$state" == exited ]] || {
  echo "Stop the dedicated container before run_all.sh: $container ($state)" >&2
  exit 2
}

if [[ -n "${MMBENCH_BASELINE_TSV:-}" ]]; then
  "$python_bin" "$script_dir/verify_inputs.py" --baseline-tsv "$MMBENCH_BASELINE_TSV"
else
  "$python_bin" "$script_dir/verify_inputs.py"
fi
mkdir -p "$run_root"

run_one() (
  set -euo pipefail
  local model="$1"
  local model_id port
  case "$model" in
    gemma) model_id=google/gemma-3-4b-it; port=19301 ;;
    qwen36) model_id=Qwen/Qwen3.6-27B; port=19303 ;;
    qwen3vl) model_id=Qwen/Qwen3-VL-8B-Instruct; port=19304 ;;
    *) echo "Unknown model key: $model" >&2; return 2 ;;
  esac
  port="${MMBENCH_PORT:-$port}"
  local run_dir="$run_root/$model"
  local endpoint="http://127.0.0.1:$port/v1/chat/completions"
  local base="http://127.0.0.1:$port"
  local service_pid=""
  local container_started=0
  mkdir -p "$run_dir" || return

  cleanup() {
    trap - EXIT
    if [[ "$container_started" == 1 ]]; then
      docker stop --time 30 "$container_id" >/dev/null 2>&1 || docker kill "$container_id" >/dev/null 2>&1 || true
    fi
    if [[ -n "$service_pid" ]]; then
      wait "$service_pid" >/dev/null 2>&1 || true
    fi
  }
  trap cleanup EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM

  local current_state
  current_state="$(docker inspect --format '{{.State.Status}}' "$container_id" 2>/dev/null || true)"
  [[ "$current_state" == created || "$current_state" == exited ]] || {
    echo "Dedicated container is unexpectedly $current_state" >&2
    return 3
  }
  if ss -lnt | awk '{print $4}' | grep -Eq ":${port}$"; then
    echo "Port already in use before starting $model: $port" >&2
    return 3
  fi
  docker start "$container_id" >/dev/null || return
  container_started=1
  docker exec -e "MMBENCH_PORT=$port" "$container_id" \
    /bin/bash /benchmark/submit/scripts/start_lightllm.sh "$model" \
    > "$run_dir/server.log" 2>&1 &
  service_pid=$!

  local ready=0
  for ((step=0; step<180; step++)); do
    if curl --noproxy '*' --silent --fail --max-time 3 "$base/health" >/dev/null \
        && [[ "$(docker inspect --format '{{.State.Running}}' "$container_id" 2>/dev/null || true)" == true ]] \
        && jobs -pr | grep -Fxq "$service_pid"; then
      ready=1
      break
    fi
    if ! jobs -pr | grep -Fxq "$service_pid"; then
      break
    fi
    sleep 5
  done
  if [[ "$ready" != 1 ]]; then
    echo "LightLLM did not become healthy for $model; see $run_dir/server.log" >&2
    return 3
  fi

  local -a reasoning=()
  if [[ "$model" == qwen36 ]]; then
    "$python_bin" "$runner" probe-reasoning \
      --url "$endpoint" \
      --model "$model_id" \
      --output "$run_dir/reasoning_probe.json" || return
    reasoning=(--reasoning-effort none)
  fi

  "$python_bin" "$runner" run \
    --url "$endpoint" \
    --model "$model_id" \
    --limit 2 \
    --output "$run_dir/precheck" \
    "${reasoning[@]}" || return
  "$python_bin" "$runner" run \
    --url "$endpoint" \
    --model "$model_id" \
    --limit 200 \
    --output "$run_dir/full" \
    "${reasoning[@]}" || return
  echo "Completed $model: $run_dir/full/summary.json"
)

status=0
for model in gemma qwen36 qwen3vl; do
  if ! run_one "$model"; then
    status=1
    echo "Failed $model; no score is reported for an incomplete request set" >&2
  fi
done
exit "$status"
