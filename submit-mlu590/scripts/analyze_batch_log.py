#!/usr/bin/env python3
"""Summarize LightLLM's observed dynamic batch sizes during one run stage."""

import argparse
import collections
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo


STAMP = re.compile(r"(?:INFO|DEBUG) (\d\d-\d\d \d\d:\d\d:\d\d) ")
CURRENT = re.compile(r"current batch size: (\d+)")
NEW_BATCH = re.compile(r"generate new batch, .*?\(req_id, shm_index\)s:(\[[^\]]*\])")
REQ = re.compile(r"\(\d+,\s*\d+\)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    local_zone = ZoneInfo("Asia/Shanghai")
    start = datetime.fromisoformat(summary["started_at_utc"]).astimezone(local_zone)
    finish = datetime.fromisoformat(summary["finished_at_utc"]).astimezone(local_zone)
    # LightLLM logs only whole seconds. Include the boundary seconds.
    start_floor = start.replace(microsecond=0)
    finish_ceil = finish.replace(microsecond=0) + timedelta(seconds=1)
    current_sizes = collections.Counter()
    new_batch_sizes = collections.Counter()
    for line in args.log.read_text(encoding="utf-8", errors="replace").splitlines():
        stamp_match = STAMP.search(line)
        if stamp_match is None:
            continue
        stamp = datetime.strptime(f"{start.year}-{stamp_match.group(1)}", "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=local_zone
        )
        if stamp < start_floor or stamp >= finish_ceil:
            continue
        current_match = CURRENT.search(line)
        if current_match:
            current_sizes[int(current_match.group(1))] += 1
        new_match = NEW_BATCH.search(line)
        if new_match:
            new_batch_sizes[len(REQ.findall(new_match.group(1)))] += 1
    result = {
        "model": summary["model"],
        "stage_questions": summary["stage_questions"],
        "window_local_start": start.isoformat(),
        "window_local_finish": finish.isoformat(),
        "log_path": str(args.log.resolve()),
        "current_batch_size_debug_samples": dict(sorted(current_sizes.items())),
        "max_observed_running_batch_size": max(current_sizes, default=None),
        "new_batch_scheduled_size_counts": dict(sorted(new_batch_sizes.items())),
        "interpretation": (
            "current_batch_size is a LightLLM router log sample every 100 scheduling steps; "
            "new_batch_scheduled_size counts newly admitted requests, before merging with the running batch. "
            "Neither graph_max_batch_size nor n=1 is the measured batch size."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
