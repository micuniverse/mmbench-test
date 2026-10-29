#!/usr/bin/env python3
"""C500 MMBench validation[:200] through VLMEvalKit and local LightLLM.

The preparation step uses VLMEvalKit's ImageMCQDataset.build_prompt and
OpenAIWrapper.prepare_inputs. Requests and blind extraction never load gold
answers. After all requests have succeeded, one CustomMCQDataset.evaluate
call scores the frozen A-D/Z predictions.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent
for dependency_dir in (
    ROOT.parent / "test-two/deps",
    ROOT.parent / "test-two/deps_compat",
    Path("/data/wangpengyu_vlm/vbench-test-c500/scoring_attempts/"
         "c500_official_vbench_20261024/venv/lib/python3.12/site-packages"),
):
    if dependency_dir.is_dir():
        sys.path.insert(0, str(dependency_dir))

import httpx
import pandas as pd
from choice_extractor import RULE_VERSION, Extraction, extract_option


PARQUET = ROOT / "dataset/data/validation-00000-of-00001-93861598ee8837c0.parquet"
PROMPT_VERSION = "c500_vlmeval_letter_only_v1"
ANSWER_INSTRUCTION = (
    "Answer with only the single uppercase option letter from the listed options "
    "(A/B/C/D as applicable). Do not provide reasoning or any other text."
)
PREPARED = ROOT / "dataset/prepared_messages_200.jsonl"
PREPARED_META = ROOT / "dataset/prepared_messages_200.meta.json"
VLMEVAL_ROOT = Path("/data/wangpengyu_vlm/frameworks/VLMEvalKit")
HF_REVISION = "215b79e126bea0777a489683f239edd0a6012a8a"
EXPECTED_PARQUET_SHA256 = "741977ca4e9badf4383a244815c03b0de2b129aac4c97b0ff71784a0c63fa14f"
EXPECTED_TSV_SHA256 = "aeac8ac1c905eec876d824ff19b9d6279903272a0b41d918c34072a437387df4"
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")


def load_vlmeval():
    """Load only the VLMEvalKit modules used here, avoiding optional backends.

    VLMEvalKit's package __init__ imports all model and dataset integrations,
    including unavailable optional video dependencies. Namespace packages
    allow the actual image MCQ, OpenAI adapter, and matching modules to load.
    """
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(VLMEVAL_ROOT))
    package_root = VLMEVAL_ROOT / "vlmeval"
    for name, directory in (
        ("vlmeval", package_root),
        ("vlmeval.dataset", package_root / "dataset"),
        ("vlmeval.api", package_root / "api"),
    ):
        module = types.ModuleType(name)
        module.__package__ = name
        module.__path__ = [str(directory)]
        sys.modules[name] = module
    from vlmeval.api.gpt import OpenAIWrapper
    from vlmeval.dataset.image_mcq import CustomMCQDataset, ImageMCQDataset
    return ImageMCQDataset, OpenAIWrapper, CustomMCQDataset


def validate_source(tsv: Path | None = None) -> dict[str, Any]:
    if not PARQUET.is_file():
        raise FileNotFoundError(f"Missing HuggingFaceM4/MMBench validation parquet: {PARQUET}")
    parquet_sha = sha256(PARQUET)
    if parquet_sha != EXPECTED_PARQUET_SHA256:
        raise ValueError(f"Parquet SHA256 mismatch: {parquet_sha}")
    tsv_check: dict[str, Any] = {"status": "not_present"}
    if tsv is not None:
        if not tsv.is_file():
            raise FileNotFoundError(tsv)
        tsv_sha = sha256(tsv)
        tsv_check = {"path": str(tsv.resolve()), "sha256": tsv_sha,
                     "matches_expected": tsv_sha == EXPECTED_TSV_SHA256}
        if tsv_sha != EXPECTED_TSV_SHA256:
            raise ValueError(f"Baseline TSV SHA256 mismatch: {tsv_sha}")
    return {
        "dataset": "HuggingFaceM4/MMBench",
        "split": "validation[:200]",
        "revision": HF_REVISION,
        "parquet_path": str(PARQUET),
        "parquet_sha256": parquet_sha,
        "baseline_tsv": tsv_check,
    }


def read_prepared() -> list[dict[str, Any]]:
    if not PREPARED.is_file() or not PREPARED_META.is_file():
        raise FileNotFoundError("Run `prepare` before `run`.")
    metadata = json.loads(PREPARED_META.read_text(encoding="utf-8"))
    if metadata.get("parquet_sha256") != EXPECTED_PARQUET_SHA256:
        raise ValueError("Prepared source hash mismatch")
    if metadata.get("prepared_sha256") != sha256(PREPARED):
        raise ValueError("Prepared messages SHA256 mismatch")
    if metadata.get("prompt_version") != PROMPT_VERSION:
        raise ValueError("Prepared prompt version mismatch")
    with PREPARED.open(encoding="utf-8") as file:
        records = [json.loads(line) for line in file if line.strip()]
    if len(records) != 200 or [row["position"] for row in records] != list(range(200)):
        raise ValueError("Prepared records are not validation rows 0 through 199")
    if any("answer" in row for row in records):
        raise ValueError("Prepared request records must not contain gold answers")
    return records


def prepare(args: argparse.Namespace) -> None:
    source = validate_source(args.tsv)
    table = pd.read_parquet(PARQUET)
    if len(table) != 4329:
        raise ValueError(f"Expected 4329 validation rows, found {len(table)}")
    dataset_class, wrapper_class, _ = load_vlmeval()
    os.environ["LMUData"] = str(ROOT / "dataset")

    class LocalMMBench(dataset_class):
        def load_data(self, dataset: str) -> pd.DataFrame:
            return table

    dataset = LocalMMBench(dataset="HF_MMBench_validation")
    # prepare_inputs performs VLMEvalKit's image encoding into OpenAI image_url.
    wrapper = wrapper_class(model="local", key="local", api_base="http://127.0.0.1/v1/chat/completions")
    records = []
    for position in range(200):
        row = dataset.data.iloc[position]
        options = {key: str(row[key]) for key in "ABCD" if key in row and not pd.isna(row[key])}
        if not options:
            raise ValueError(f"Row {position}: no options")
        built_prompt = dataset.build_prompt(position)
        messages = wrapper.prepare_inputs(built_prompt)
        if len(messages) != 1 or messages[0].get("role") != "user":
            raise ValueError(f"Row {position}: unexpected OpenAI messages")
        parts = messages[0].get("content", [])
        if not any(part.get("type") == "image_url" for part in parts):
            raise ValueError(f"Row {position}: missing image_url")
        if not any(part.get("type") == "text" for part in parts):
            raise ValueError(f"Row {position}: missing text")
        # Preserve VLMEvalKit's question, options, and image encoding while
        # making the expected answer format explicit for this new request run.
        text_parts = [part for part in parts if part.get("type") == "text"]
        if len(text_parts) != 1 or not isinstance(text_parts[0].get("text"), str):
            raise ValueError(f"Row {position}: unexpected text message structure")
        text_parts[0]["text"] = text_parts[0]["text"].rstrip() + "\n" + ANSWER_INSTRUCTION
        records.append({
            "position": position,
            "index": int(row["index"]),
            "question": str(row["question"]),
            "options": options,
            "category": None if pd.isna(row.get("category")) else str(row["category"]),
            "l2_category": None if pd.isna(row.get("l2-category")) else str(row["l2-category"]),
            "messages": messages,
        })
    PREPARED.parent.mkdir(parents=True, exist_ok=True)
    with PREPARED.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    metadata = {**source, "validation_rows": len(table), "prepared_rows": len(records),
                "prepared_path": str(PREPARED), "prepared_sha256": sha256(PREPARED),
                "prompt_version": PROMPT_VERSION, "answer_instruction": ANSWER_INSTRUCTION,
                "vlmeval_root": str(VLMEVAL_ROOT), "prepared_at_utc": utc_now()}
    write_json(PREPARED_META, metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


async def request_one(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    url: str,
    record: dict[str, Any],
    payload: dict[str, Any],
    output_file,
    output_lock: asyncio.Lock,
) -> dict[str, Any]:
    async with semaphore:
        started_at = utc_now()
        start = time.perf_counter()
        result: dict[str, Any] = {
            "position": record["position"], "index": record["index"],
            "request_started_at_utc": started_at, "request": payload,
            "http_status": None, "response_text": None, "transport_error": None,
        }
        try:
            response = await client.post(url, json=payload)
            result["http_status"] = response.status_code
            result["response_text"] = response.text
        except Exception as exc:
            result["transport_error"] = f"{type(exc).__name__}: {exc}"
        result["request_finished_at_utc"] = utc_now()
        result["request_elapsed_s"] = round(time.perf_counter() - start, 6)
        async with output_lock:
            output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            output_file.flush()
        return result


def audit_one(record: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    """Parse one API response, with no access to the gold answer during extraction."""
    raw_reply: str | None = None
    finish_reason: str | None = None
    response_error: str | None = None
    if response.get("transport_error"):
        response_error = str(response["transport_error"])
    elif response.get("http_status") != 200:
        response_error = f"HTTP {response.get('http_status')}"
    else:
        try:
            body = json.loads(response["response_text"])
            choices = body["choices"]
            if not isinstance(choices, list) or len(choices) != 1:
                raise ValueError("Expected exactly one choice")
            choice = choices[0]
            finish_reason = choice.get("finish_reason")
            raw_reply = choice["message"]["content"]
            if not isinstance(raw_reply, str):
                raise ValueError("Assistant content is not text")
            if finish_reason not in ("stop", "length"):
                raise ValueError(f"Unexpected finish_reason={finish_reason!r}")
        except (KeyError, TypeError, ValueError) as exc:
            response_error = f"Malformed API response: {type(exc).__name__}: {exc}"

    extraction = (Extraction("Z", "request_error", response_error)
                  if response_error else extract_option(raw_reply, finish_reason, record["options"],
                                                        question=record["question"]))
    status = ("request_error" if response_error else
              "truncated" if finish_reason == "length" else
              "unresolved" if extraction.option == "Z" else "parsed")
    return {
        "extraction_rule_version": RULE_VERSION,
        "position": record["position"], "index": record["index"],
        "question": record["question"],
        "options": record["options"],
        "request_elapsed_s": response.get("request_elapsed_s"),
        "http_status": response.get("http_status"),
        "transport_error": response.get("transport_error"),
        "raw_reply": raw_reply, "finish_reason": finish_reason,
        "extracted_option": extraction.option, "extraction_rule": extraction.rule,
        "extraction_evidence": extraction.evidence,
        "explicit_claims": list(extraction.explicit_claims),
        "response_error": response_error, "status": status, "hit": None,
    }


def validate_response_set(records: list[dict[str, Any]],
                          responses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Refuse missing, duplicated, or mixed responses before creating a score."""
    expected = {row["position"]: row["index"] for row in records}
    record_by_position = {row["position"]: row for row in records}
    if (len(expected) != len(records) or len(responses) != len(records)
            or len(set(expected.values())) != len(records)):
        raise ValueError("The response count or prepared positions are not unique")
    by_position: dict[int, dict[str, Any]] = {}
    models: set[str] = set()
    reasoning_efforts: set[str | None] = set()
    for response in responses:
        position = response.get("position")
        if position in by_position or position not in expected:
            raise ValueError(f"Unexpected or duplicate response position: {position!r}")
        if response.get("index") != expected[position]:
            raise ValueError(f"Response index mismatch at position {position!r}")
        request = response.get("request")
        if not isinstance(request, dict) or request.get("messages") != record_by_position[position]["messages"]:
            raise ValueError(f"Request prompt mismatch at position {position!r}")
        if (request.get("n") != 1 or request.get("temperature") != 0
                or request.get("max_tokens") != 1024):
            raise ValueError(f"Request sampling parameters mismatch at position {position!r}")
        models.add(str(request.get("model")))
        reasoning_efforts.add(request.get("reasoning_effort"))
        by_position[position] = response
    if set(by_position) != set(expected):
        raise ValueError("Missing response positions")
    if len(models) != 1 or "None" in models:
        raise ValueError("Mixed or missing model IDs in the response set")
    if len(reasoning_efforts) != 1:
        raise ValueError("Mixed reasoning_effort values in the response set")
    return [by_position[row["position"]] for row in records]


def score_responses(records: list[dict[str, Any]], responses: list[dict[str, Any]],
                    output: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    """Normalize to A-D/Z, then score exclusively through CustomMCQDataset."""
    score_start = time.perf_counter()
    responses = validate_response_set(records, responses)
    audits = [audit_one(row, response) for row, response in zip(records, responses)]
    request_errors = [audit["index"] for audit in audits if audit["status"] == "request_error"]
    if request_errors:
        raise RuntimeError(f"Request failures prevent scoring; indices={request_errors}")
    # Freeze the gold-blind extraction before opening the benchmark answer column.
    extraction_path = output / "blind_extraction.jsonl"
    with extraction_path.open("x", encoding="utf-8") as file:
        for audit in audits:
            file.write(json.dumps(audit, ensure_ascii=False) + "\n")
    extraction_sha = sha256(extraction_path)

    if sha256(PARQUET) != EXPECTED_PARQUET_SHA256:
        raise ValueError("Benchmark parquet changed before scoring")
    source_rows = pd.read_parquet(PARQUET).iloc[:len(records)]
    if (len(source_rows) != len(records)
            or source_rows["index"].astype(int).tolist() != [row["index"] for row in records]):
        raise ValueError("Benchmark source order changed before scoring")
    dataset_name = f"HF_MC_VAL{len(records)}"  # No 'mmbench': that selects circular evaluation.
    dataset_path = output / f"{dataset_name}.tsv"
    predictions_path = output / "normalized_predictions.tsv"
    audit_path = output / "audit.jsonl"

    dataset_rows = []
    prediction_rows = []
    for position, (record, audit) in enumerate(zip(records, audits)):
        common = {"index": record["index"], "question": record["question"],
                  **record["options"], "category": record["category"],
                  "l2-category": record["l2_category"]}
        answer = str(source_rows.iloc[position]["answer"]).strip().upper()
        if answer not in record["options"]:
            raise ValueError(f"Invalid benchmark answer at position {position}")
        dataset_rows.append({**common, "answer": answer})
        prediction_rows.append({**common, "prediction": audit["extracted_option"]})
    pd.DataFrame(dataset_rows).to_csv(dataset_path, sep="\t", index=False)
    pd.DataFrame(prediction_rows).to_csv(predictions_path, sep="\t", index=False)

    _, _, custom_dataset_class = load_vlmeval()
    old_lmu = os.environ.get("LMUData")
    old_pred = os.environ.get("PRED_FORMAT")
    old_eval = os.environ.get("EVAL_FORMAT")
    os.environ["LMUData"] = str(output)
    os.environ["PRED_FORMAT"] = "tsv"
    os.environ["EVAL_FORMAT"] = "csv"
    try:
        dataset = custom_dataset_class(dataset=dataset_name, skip_noimg=False)
        if len(dataset.data) != len(records):
            raise ValueError("VLMEvalKit metadata row count changed")
        score_frame = dataset.evaluate(str(predictions_path), model="exact_matching", nproc=1)
    finally:
        if old_lmu is None:
            os.environ.pop("LMUData", None)
        else:
            os.environ["LMUData"] = old_lmu
        if old_pred is None:
            os.environ.pop("PRED_FORMAT", None)
        else:
            os.environ["PRED_FORMAT"] = old_pred
        if old_eval is None:
            os.environ.pop("EVAL_FORMAT", None)
        else:
            os.environ["EVAL_FORMAT"] = old_eval

    result_path = output / "normalized_predictions_exact_matching_result.tsv"
    score_path = output / "normalized_predictions_acc.csv"
    if not result_path.is_file() or not score_path.is_file():
        raise FileNotFoundError("VLMEvalKit did not produce its result and score files")
    result = pd.read_csv(result_path, sep="\t")
    if (len(result) != len(records) or result["index"].nunique() != len(records)
            or set(result["index"].astype(int)) != {row["index"] for row in records}):
        raise ValueError("VLMEvalKit result did not retain every question")
    hit_by_index = {int(row["index"]): int(row["hit"]) for _, row in result.iterrows()}
    if set(hit_by_index.values()) - {0, 1}:
        raise ValueError("VLMEvalKit produced an invalid per-question hit")
    correct = sum(hit_by_index.values())
    if len(score_frame) != 1 or "Overall" not in score_frame:
        raise ValueError("VLMEvalKit did not return a single Overall score")
    accuracy = float(score_frame.iloc[0]["Overall"])
    if abs(accuracy - correct / len(records)) > 1e-12:
        raise ValueError("VLMEvalKit Overall and per-question hits disagree")
    for audit in audits:
        audit["hit"] = hit_by_index[audit["index"]]
        if audit["extracted_option"] == "Z" and audit["hit"] != 0:
            raise ValueError(f"VLMEvalKit did not count Z as wrong at index {audit['index']}")
    with audit_path.open("w", encoding="utf-8") as file:
        for audit in audits:
            file.write(json.dumps(audit, ensure_ascii=False) + "\n")

    def positions(status: str) -> tuple[list[int], list[int]]:
        rows = [row for row in audits if row["status"] == status]
        return [row["position"] for row in rows], [row["index"] for row in rows]

    unresolved_positions, unresolved_indices = positions("unresolved")
    truncated_positions, truncated_indices = positions("truncated")
    incomplete_positions, incomplete_indices = positions("incomplete")
    error_positions, error_indices = positions("request_error")
    return {
        "status": "scored", "scoring_method": "CustomMCQDataset.evaluate(model='exact_matching', nproc=1)",
        "prompt_version": metadata["prompt_version"],
        "extraction_rule_version": RULE_VERSION,
        "extraction_rule_sha256": sha256(SCRIPT_DIR / "choice_extractor.py"),
        "blind_extraction": str(extraction_path),
        "blind_extraction_sha256": extraction_sha,
        "stage_questions": len(records), "denominator": len(records),
        "correct": correct, "accuracy": accuracy,
        "unresolved_count": len(unresolved_indices), "unresolved_positions": unresolved_positions,
        "unresolved_indices": unresolved_indices,
        "truncated_count": len(truncated_indices), "truncated_positions": truncated_positions,
        "truncated_indices": truncated_indices,
        "incomplete_count": len(incomplete_indices), "incomplete_positions": incomplete_positions,
        "incomplete_indices": incomplete_indices,
        "request_error_count": len(error_indices), "request_error_positions": error_positions,
        "request_error_indices": error_indices,
        "normalized_predictions": str(predictions_path), "evaluation_dataset_tsv": str(dataset_path),
        "vlmeval_result": str(result_path), "vlmeval_score": str(score_path),
        "audit": str(audit_path), "prepared_messages_sha256": metadata["prepared_sha256"],
        "scoring_and_audit_s": round(time.perf_counter() - score_start, 6),
    }


async def run_async(args: argparse.Namespace) -> None:
    records = read_prepared()[: args.limit]
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Use a fresh output directory for this request run: {output}")
    output.mkdir(parents=True)
    raw_path = output / "raw_requests_responses.jsonl"
    summary_path = output / "summary.json"
    url = args.url.rstrip("/")
    if not url.endswith("/v1/chat/completions"):
        raise ValueError("--url must end in /v1/chat/completions")
    payloads = []
    for record in records:
        payload: dict[str, Any] = {
            "model": args.model, "messages": record["messages"],
            "n": 1, "temperature": 0, "max_tokens": 1024,
        }
        if args.reasoning_effort is not None:
            payload["reasoning_effort"] = args.reasoning_effort
        payloads.append(payload)
    metadata = json.loads(PREPARED_META.read_text(encoding="utf-8"))
    started_at = utc_now()
    total_start = time.perf_counter()
    semaphore = asyncio.Semaphore(4)
    output_lock = asyncio.Lock()
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(args.timeout, connect=20.0), trust_env=False,
        limits=httpx.Limits(max_connections=4, max_keepalive_connections=4),
    ) as client:
        with raw_path.open("w", encoding="utf-8") as raw_file:
            tasks = [request_one(client, semaphore, url, row, payload, raw_file, output_lock)
                     for row, payload in zip(records, payloads)]
            responses = await asyncio.gather(*tasks)
    request_phase_s = time.perf_counter() - total_start
    try:
        score_info = score_responses(records, responses, output, metadata)
    except Exception as exc:
        write_json(summary_path, {
            "status": "not_scored", "reason": f"{type(exc).__name__}: {exc}",
            "model": args.model, "endpoint": url, "stage_questions": args.limit,
            "prompt_version": metadata["prompt_version"],
            "raw_requests_responses": str(raw_path),
            "raw_requests_responses_sha256": sha256(raw_path),
            "started_at_utc": started_at, "finished_at_utc": utc_now(),
            "request_phase_s": round(request_phase_s, 6),
        })
        raise
    total_s = time.perf_counter() - total_start
    summary = {
        **score_info,
        "run_kind": "online_requests_and_scoring",
        "model": args.model, "endpoint": url, "stage_questions": args.limit,
        "dataset": metadata["dataset"], "split": f"validation[:{args.limit}]",
        "dataset_revision": metadata["revision"],
        "parquet_sha256": metadata["parquet_sha256"],
        "prepared_messages_sha256": metadata["prepared_sha256"],
        "prompt_version": metadata["prompt_version"],
        "baseline_tsv": metadata["baseline_tsv"],
        "request_parameters": {"n": 1, "temperature": 0, "max_tokens": 1024,
                               "concurrency": 4, "reasoning_effort": args.reasoning_effort},
        "requests_completed": len(responses),
        "started_at_utc": started_at, "finished_at_utc": utc_now(),
        "request_phase_s": round(request_phase_s, 6),
        "total_requests_and_scoring_s": round(total_s, 6),
        "timing_scope": "Starts before request dispatch; ends after normalization, VLMEvalKit evaluation, and audit writing.",
        "raw_requests_responses": str(raw_path), "raw_requests_responses_sha256": sha256(raw_path),
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def probe_reasoning(args: argparse.Namespace) -> None:
    """Confirm the active endpoint accepts Qwen3.6 reasoning_effort=none."""
    url = args.url.rstrip("/")
    if not url.endswith("/v1/chat/completions"):
        raise ValueError("--url must end in /v1/chat/completions")
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": "Reply with exactly OK."}],
        "n": 1, "temperature": 0, "max_tokens": 16,
        "reasoning_effort": "none",
    }
    started_at = utc_now()
    start = time.perf_counter()
    with httpx.Client(timeout=httpx.Timeout(args.timeout, connect=20.0), trust_env=False) as client:
        response = client.post(url, json=payload)
    elapsed = time.perf_counter() - start
    try:
        parsed = response.json()
    except ValueError:
        parsed = None
    choice = ((parsed or {}).get("choices") or [None])[0]
    message = choice.get("message", {}) if isinstance(choice, dict) else {}
    accepted = (response.status_code == 200 and isinstance(choice, dict)
                and isinstance(message.get("content"), str)
                and not message.get("reasoning_content")
                and choice.get("finish_reason") == "stop")
    artifact = {
        "accepted": accepted, "started_at_utc": started_at,
        "finished_at_utc": utc_now(), "elapsed_s": round(elapsed, 6),
        "endpoint": url, "request": payload,
        "http_status": response.status_code, "response_text": response.text,
        "assistant_content": message.get("content"),
        "reasoning_content": message.get("reasoning_content"),
        "finish_reason": choice.get("finish_reason") if isinstance(choice, dict) else None,
    }
    write_json(args.output.resolve(), artifact)
    print(json.dumps({key: value for key, value in artifact.items() if key != "response_text"},
                     ensure_ascii=False, indent=2))
    if not accepted:
        raise RuntimeError("Active endpoint did not accept reasoning_effort=none")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare", help="Build validation[:200] VLMEvalKit OpenAI messages")
    prepare_parser.add_argument("--tsv", type=Path, help="Optional local benchmark TSV; verify SHA256")
    run_parser = commands.add_parser("run", help="Run 2 or 200 requests against local LightLLM")
    run_parser.add_argument("--url", required=True, help="Full local /v1/chat/completions URL")
    run_parser.add_argument("--model", required=True, help="Model ID in the OpenAI request")
    run_parser.add_argument("--limit", required=True, type=int, choices=(2, 200))
    run_parser.add_argument("--output", required=True, type=Path)
    run_parser.add_argument("--reasoning-effort", choices=("none",))
    run_parser.add_argument("--timeout", type=float, default=600.0)
    probe_parser = commands.add_parser("probe-reasoning", help="Confirm live Qwen3.6 endpoint accepts none")
    probe_parser.add_argument("--url", required=True)
    probe_parser.add_argument("--model", required=True)
    probe_parser.add_argument("--output", required=True, type=Path)
    probe_parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "run":
        asyncio.run(run_async(args))
    else:
        probe_reasoning(args)


if __name__ == "__main__":
    main()
