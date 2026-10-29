#!/usr/bin/env python3
"""Run local LightLLM inference and MCQ scoring with VLMEvalKit source code.

The framework builds each image/question prompt, sends the OpenAI-compatible
request through LMDeployAPI, and scores the resulting A-D predictions with
CustomMCQDataset.evaluate(exact_matching). Raw HTTP exchanges are saved.
Unparseable replies or request failures stop scoring instead of becoming
silent format-related wrong answers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
import re
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests


WORK = Path(__file__).resolve().parent.parent.parent
PROJECT = WORK.parent
VLMEVAL = PROJECT / "frameworks" / "VLMEvalKit" / "vlmeval"
SOURCE_TSV = PROJECT / "data" / "vlmeval" / "HFM4_MCQs_VAL_200.tsv"
LOCAL_TSV = WORK / "data" / SOURCE_TSV.name
DATASET_NAME = LOCAL_TSV.stem
FORMAT_RULE = (
    "\nAnswer with exactly one uppercase option letter (A, B, C, or D). "
    "Do not include words, punctuation, or an explanation.\n"
)
OUTPUT_FIELDS = ["index", "question", "prediction", "A", "B", "C", "D",
                 "answer", "split", "category", "l2-category", "source_row_idx"]


# Exact copies of the two functions used by the original run in run_two_cases.py.
def parse_option(content: str, options: list[str]) -> str | None:
    text = content.strip()
    exact = re.fullmatch(r"(?:[（(]?\s*)([A-D])(?:\s*[）).。]?\s*)", text, flags=re.I)
    if exact and exact.group(1).upper() in options:
        return exact.group(1).upper()
    leading = re.match(r"^[*_`]*\s*[（(]?\s*([A-D])\s*[.)、:：]", text, flags=re.I)
    if leading and leading.group(1).upper() in options:
        return leading.group(1).upper()
    # Prefer a stated answer over option names mentioned later in an explanation.
    for label in (r"final\s+answer|correct\s+answer|答案", r"answer", r"choice|option|选项"):
        explicit = re.findall(
            rf"(?:{label})[\s*_`]*(?:is|为|是)?[\s*_`]*[:：]?[\s*_`]*[（(]?[\s*_`]*([A-D])\b",
            text, flags=re.I,
        )
        candidates = {candidate.upper() for candidate in explicit if candidate.upper() in options}
        if len(candidates) == 1:
            return candidates.pop()
        if candidates:
            return None
    # Keep ambiguous and unparseable generations visible as incorrect.
    return None


def response_content(response: dict) -> str:
    if response.get("object") != "chat.completion":
        raise ValueError("POST /v1/chat/completions did not return an OpenAI chat completion")
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("Chat completion has no choices[0]")
    if choices[0].get("finish_reason") == "length":
        raise ValueError("Chat completion was truncated at max_tokens")
    message = choices[0].get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise ValueError("Chat completion has no assistant message")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Chat completion has no text content")
    return content


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def load_framework():
    """Load the original relevant modules without eager imports of other benchmarks."""
    for dep in (PROJECT / ".eval_deps", WORK / ".deps"):
        if dep.is_dir():
            sys.path.insert(0, str(dep))
    sys.path.insert(0, str(PROJECT))
    for name, path in (
        ("vlmeval", VLMEVAL),
        ("vlmeval.dataset", VLMEVAL / "dataset"),
        ("vlmeval.dataset.utils", VLMEVAL / "dataset" / "utils"),
        ("vlmeval.api", VLMEVAL / "api"),
    ):
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    dataset_utils = sys.modules["vlmeval.dataset.utils"]
    dataset_utils.DEBUG_MESSAGE = "External judge disabled for this run"

    def no_judge(**_kwargs):
        raise RuntimeError("External judge must not be called")

    dataset_utils.build_judge = no_judge
    # LMDeployAPI imports this registry even when no adapter is selected.
    adapters = types.ModuleType("vlmeval.api.adapters")
    adapters.build_adapter = lambda _name: (_ for _ in ()).throw(
        RuntimeError("No framework adapter is selected in this runner"))
    sys.modules[adapters.__name__] = adapters

    from vlmeval.api.lmdeploy import LMDeployAPI
    from vlmeval.dataset.image_mcq import CustomMCQDataset
    from vlmeval.smp import get_intermediate_file_path, load
    return parse_option, response_content, LMDeployAPI, CustomMCQDataset, get_intermediate_file_path, load


def validate_base_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.path != "/v1":
        raise ValueError("--base-url must be an http(s) URL ending exactly in /v1")
    return base_url


def bypass_proxy_for_loopback() -> None:
    # requests.get below and VLMEvalKit's requests.post both honor these.
    for name in ("NO_PROXY", "no_proxy"):
        existing = [entry.strip() for entry in os.environ.get(name, "").split(",") if entry.strip()]
        for host in ("127.0.0.1", "localhost", "::1"):
            if host not in existing:
                existing.append(host)
        os.environ[name] = ",".join(existing)


def get_model_id(base_url: str, requested: str | None, timeout: int) -> str:
    reply = requests.get(f"{base_url}/models", timeout=min(timeout, 30))
    reply.raise_for_status()
    data = reply.json()
    if data.get("object") != "list" or not isinstance(data.get("data"), list):
        raise ValueError("GET /v1/models returned an invalid OpenAI-compatible model list")
    ids = [entry.get("id") for entry in data["data"] if isinstance(entry, dict)]
    ids = [item for item in ids if isinstance(item, str) and item]
    if requested:
        if requested not in ids:
            raise ValueError(f"Requested model {requested!r} is not in /v1/models: {ids}")
        return requested
    if len(ids) != 1:
        raise ValueError(f"Expected one model from /v1/models, found {ids}; pass --model")
    return ids[0]


def options_for(row) -> list[str]:
    return [key for key in "ABCD"
            if key in row and isinstance(row[key], str) and row[key].strip()]


def add_format_rule(prompt: list[dict]) -> tuple[list[dict], str]:
    prompt = [dict(item) for item in prompt]
    texts = [item for item in prompt if item.get("type") == "text"]
    if not texts:
        raise ValueError("VLMEvalKit prompt has no text part")
    texts[-1]["value"] += FORMAT_RULE
    return prompt, texts[-1]["value"]


def request_from_response(response) -> dict:
    body = response.request.body
    if isinstance(body, bytes):
        body = body.decode("utf-8")
    return json.loads(body)


def case_prefix(position: int, index: int) -> str:
    return f"case_{position:03d}_{index}"


def reusable_request(request_data: dict, model_id: str, prompt_text: str,
                     max_tokens: int, extra_body: dict) -> bool:
    if not isinstance(request_data, dict):
        return False
    if (request_data.get("model") != model_id
            or request_data.get("max_tokens") != max_tokens
            or request_data.get("temperature") != 0):
        return False
    if any(request_data.get(key) != value for key, value in extra_body.items()):
        return False
    messages = request_data.get("messages")
    if not isinstance(messages, list) or len(messages) != 1:
        return False
    user = messages[0]
    if user.get("role") != "user" or not isinstance(user.get("content"), list):
        return False
    texts = [part.get("text") for part in user["content"]
             if isinstance(part, dict) and part.get("type") == "text"]
    return texts == [prompt_text]


def build_audit(position: int, index: int, response_data: dict, request_data: dict,
                options: list[str], prompt_text: str, response_file: Path,
                request_file: Path, parse_option, response_content,
                reused: bool) -> dict:
    content = response_content(response_data)
    choice = parse_option(content, options)
    return {
        "source_row_idx": position,
        "dataset_index": index,
        "status": "parsed" if choice else "unparsed",
        "valid_options": options,
        "raw_content": content,
        "canonical_prediction": choice,
        "finish_reason": response_data["choices"][0].get("finish_reason"),
        "model_id": request_data.get("model"),
        "prompt_text": prompt_text,
        "request_file": str(request_file),
        "request_sha256": sha256_file(request_file),
        "response_file": str(response_file),
        "response_sha256": sha256_file(response_file),
        "reused_saved_response": reused,
        "model_request_attempted": not reused,
        "http_response_observed": not reused,
    }


def main() -> int:
    run_started_at_utc = datetime.now(timezone.utc).isoformat()
    run_started_clock = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="Local LightLLM URL ending in /v1")
    parser.add_argument("--model", help="Model ID; must appear in GET /v1/models")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, choices=range(1, 201), metavar="1..200", default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--extra-body", default="{}", help="Additional JSON payload, e.g. reasoning_effort")
    parser.add_argument("--resume", action="store_true", help="Reuse valid saved HTTP responses")
    args = parser.parse_args()
    if args.workers < 1 or args.max_tokens < 1 or args.timeout < 1:
        parser.error("--workers, --max-tokens and --timeout must be positive")
    bypass_proxy_for_loopback()
    try:
        base_url = validate_base_url(args.base_url)
        extra_body = json.loads(args.extra_body)
        if not isinstance(extra_body, dict):
            raise ValueError("--extra-body must be a JSON object")
        forbidden = {"model", "messages", "n", "max_tokens", "temperature", "stream",
                     "api_base", "key", "timeout", "retry", "wait", "verbose"}
        if set(extra_body) & forbidden:
            raise ValueError(f"--extra-body overrides reserved keys: {sorted(set(extra_body) & forbidden)}")
        output_dir = args.output_dir.resolve()
        if not output_dir.is_relative_to(WORK):
            raise ValueError("--output-dir must stay inside VLMEvalKit-test")
        if not SOURCE_TSV.is_file() or not LOCAL_TSV.is_file():
            raise ValueError("Both source and VLMEvalKit-test/data dataset TSV files are required")
        source_sha = sha256_file(SOURCE_TSV)
        local_sha = sha256_file(LOCAL_TSV)
        if local_sha != source_sha:
            raise ValueError("VLMEvalKit-test dataset copy differs from source TSV")
        if output_dir.exists() and any(output_dir.iterdir()):
            if not args.resume:
                raise FileExistsError(f"Nonempty output directory requires --resume: {output_dir}")
            if not (output_dir / "run_config.json").is_file():
                raise ValueError("Cannot resume a nonempty output without run_config.json")
        model_id = get_model_id(base_url, args.model, args.timeout)
    except (ValueError, OSError, requests.RequestException, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    os.environ["LMUData"] = str(LOCAL_TSV.parent)
    os.environ["PRED_FORMAT"] = "tsv"
    parse_option, response_content, LMDeployAPI, CustomMCQDataset, get_intermediate_file_path, load = load_framework()
    parser_sha = sha256_bytes(inspect.getsource(parse_option).encode("utf-8"))
    dataset = CustomMCQDataset(dataset=DATASET_NAME)
    if len(dataset) < args.limit:
        raise ValueError(f"Dataset has only {len(dataset)} rows")

    config = {
        "dataset": DATASET_NAME,
        "dataset_tsv": str(LOCAL_TSV),
        "dataset_sha256": local_sha,
        "dataset_source_tsv": str(SOURCE_TSV),
        "dataset_source_sha256": source_sha,
        "base_url": base_url,
        "model_id": model_id,
        "limit": args.limit,
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "extra_body": extra_body,
        "format_rule": FORMAT_RULE,
        "parser": "run_two_cases.parse_option",
        "parser_sha256": parser_sha,
        "request_sender": "VLMEvalKit LMDeployAPI.generate -> OpenAISDKWrapper.generate_inner",
        "scorer": "VLMEvalKit CustomMCQDataset.evaluate(model='exact_matching', nproc=1)",
        "judge_api_calls": 0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    config_file = output_dir / "run_config.json"
    if config_file.exists():
        if json.loads(config_file.read_text(encoding="utf-8")) != config:
            raise ValueError("Resume configuration differs from saved run_config.json")
    else:
        write_json(config_file, config)

    class CapturingLMDeployAPI(LMDeployAPI):
        def generate_inner(self, *inner_args, **inner_kwargs):
            ret = super().generate_inner(*inner_args, **inner_kwargs)
            self.last_http_response = ret[2]
            return ret

    worker_state = threading.local()

    def worker_model():
        if not hasattr(worker_state, "model"):
            worker_state.model = CapturingLMDeployAPI(
                model=model_id, api_base=f"{base_url}/chat/completions",
                key="sk-local", max_tokens=args.max_tokens, temperature=0,
                timeout=args.timeout, retry=1, wait=0, verbose=False,
                stream=False, **extra_body)
        return worker_state.model

    tasks = []
    for position in range(args.limit):
        row = dataset.data.iloc[position]
        index = int(row["index"])
        options = options_for(row)
        if not options or row["answer"] not in options:
            raise ValueError(f"Invalid options/answer at dataset index {index}")
        prompt, prompt_text = add_format_rule(dataset.build_prompt(row))
        tasks.append((position, index, options, prompt, prompt_text))

    def run_case(task):
        position, index, options, prompt, prompt_text = task
        stem = case_prefix(position, index)
        request_file = output_dir / "cases" / f"{stem}_request.json"
        response_file = output_dir / "cases" / f"{stem}_response.json"
        audit_file = output_dir / "cases" / f"{stem}_audit.json"
        if args.resume and request_file.is_file() and response_file.is_file():
            try:
                request_data = json.loads(request_file.read_text(encoding="utf-8"))
                response_data = json.loads(response_file.read_text(encoding="utf-8"))
                if not reusable_request(request_data, model_id, prompt_text,
                                        args.max_tokens, extra_body):
                    raise ValueError("Saved request differs from this run")
                audit = build_audit(position, index, response_data, request_data, options,
                                    prompt_text, response_file, request_file,
                                    parse_option, response_content, True)
                write_json(audit_file, audit)
                return audit
            except (OSError, KeyError, ValueError, TypeError, AttributeError, json.JSONDecodeError):
                pass
        model = worker_model()
        model.last_http_response = None
        try:
            answer = model.generate(prompt, dataset=DATASET_NAME)
            response = model.last_http_response
            if response is None:
                raise RuntimeError(f"LMDeployAPI returned no HTTP response: {answer!r}")
            request_data = request_from_response(response)
            response_data = response.json()
            if response.status_code < 200 or response.status_code >= 300:
                raise RuntimeError(f"HTTP {response.status_code}: {str(response_data)[:500]}")
            content = response_content(response_data)
            if not isinstance(answer, str) or answer != content.strip():
                raise RuntimeError("LMDeployAPI returned content different from HTTP response")
            write_json(request_file, request_data)
            write_json(response_file, response_data)
            audit = build_audit(position, index, response_data, request_data, options,
                                prompt_text, response_file, request_file,
                                parse_option, response_content, False)
            write_json(audit_file, audit)
            return audit
        except Exception as exc:
            error = {
                "source_row_idx": position,
                "dataset_index": index,
                "status": "request_error",
                "model_request_attempted": True,
                "http_response_observed": False,
                "error": f"{type(exc).__name__}: {exc}",
                "prompt_text": prompt_text,
                "valid_options": options,
            }
            response = getattr(model, "last_http_response", None)
            if response is not None:
                error["http_response_observed"] = True
                error["http_status"] = response.status_code
                error["http_response_text"] = response.text
                try:
                    error["request_payload"] = request_from_response(response)
                except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
                    pass
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            write_json(output_dir / "errors" / f"{stem}_{stamp}.json", error)
            return error

    audits = []
    inference_started_clock = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_case, task): task for task in tasks}
        for number, future in enumerate(as_completed(futures), start=1):
            audit = future.result()
            audits.append(audit)
            print(f"[{number}/{len(tasks)}] index={audit['dataset_index']} "
                  f"status={audit['status']} choice={audit.get('canonical_prediction')}", flush=True)
    inference_elapsed_seconds = time.perf_counter() - inference_started_clock
    audits.sort(key=lambda item: item["source_row_idx"])
    errors = [item for item in audits if item["status"] == "request_error"]
    unparsed = [item for item in audits if item["status"] == "unparsed"]
    run_summary = {
        **config,
        "status": "needs_review" if errors or unparsed else "scored",
        "total": len(audits),
        "parsed_count": len(audits) - len(errors) - len(unparsed),
        "unparsed_count": len(unparsed),
        "request_error_count": len(errors),
        "fresh_model_request_attempts": sum(item.get("model_request_attempted", False) for item in audits),
        "fresh_model_requests_with_http_response": sum(item.get("http_response_observed", False) for item in audits),
        "reused_saved_responses": sum(item.get("reused_saved_response", False) for item in audits),
        "unparsed_indices": [item["dataset_index"] for item in unparsed],
        "request_error_indices": [item["dataset_index"] for item in errors],
        "score_valid": not (errors or unparsed),
        "normalization_uses_gold_answer": False,
        "run_started_at_utc": run_started_at_utc,
        "inference_elapsed_seconds": inference_elapsed_seconds,
    }
    if errors or unparsed:
        run_summary["total_elapsed_seconds"] = time.perf_counter() - run_started_clock
        write_json(output_dir / "run_summary.json", run_summary)
        print("Scoring withheld: inspect request errors and unparsed responses in output directory.", flush=True)
        return 2

    prediction_file = output_dir / "predictions.tsv"
    with prediction_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        for audit in audits:
            row = dataset.data.iloc[audit["source_row_idx"]]
            writer.writerow({**{key: row.get(key, "") for key in OUTPUT_FIELDS},
                             "prediction": audit["canonical_prediction"]})
    scoring_started_clock = time.perf_counter()
    accuracy_table = dataset.evaluate(str(prediction_file), model="exact_matching", nproc=1)
    result_file = Path(get_intermediate_file_path(str(prediction_file),
                                                  "_exact_matching_result"))
    result = load(str(result_file))
    hits = [int(value) for value in result["hit"]]
    if len(hits) != len(audits):
        raise ValueError("VLMEvalKit result row count differs from inference audit")
    run_summary.update({
        "correct": sum(hits),
        "accuracy": sum(hits) / len(hits),
        "prediction_file": str(prediction_file),
        "prediction_sha256": sha256_file(prediction_file),
        "result_file": str(result_file),
        "accuracy_file": str(output_dir / "predictions_acc.csv"),
        "accuracy_by_split": accuracy_table.to_dict(orient="records"),
        "scoring_elapsed_seconds": time.perf_counter() - scoring_started_clock,
    })
    run_summary["total_elapsed_seconds"] = time.perf_counter() - run_started_clock
    write_json(output_dir / "run_summary.json", run_summary)
    print(f"Framework inference + scoring: {sum(hits)}/{len(hits)} = {sum(hits) / len(hits):.1%}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
