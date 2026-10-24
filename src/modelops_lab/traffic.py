"""Deterministic traffic replay against the inference API.

Rows are selected from a verified dataset split with a seeded generator, so
the same ``(dataset, split, count, seed)`` always sends the same requests in
the same order. Every prediction's returned ``event_id`` is used to post the
delayed label when ``with_labels`` is set.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import httpx
import numpy as np

from modelops_lab.data import LoadedDataset, load_prepared_dataset, split_frame

TrafficSplit = Literal["train", "validation", "test"]
DEFAULT_TIMEOUT_SECONDS = 15.0


class TrafficError(RuntimeError):
    """Raised when the API rejects a simulated request."""


@dataclass(frozen=True)
class SimulationSummary:
    sent: int
    labeled: int
    model_versions: dict[str, int] = field(default_factory=dict)
    labels_skipped: int = 0


def _payload(dataset: LoadedDataset, row: Any) -> dict[str, float | str]:
    numeric = set(dataset.schema.numeric_features)
    payload: dict[str, float | str] = {}
    for name in dataset.schema.feature_names:
        value = row[name]
        payload[name] = float(value) if name in numeric else str(value)
    return payload


def deterministic_indices(length: int, count: int, seed: int) -> list[int]:
    """Pick ``count`` row positions reproducibly, with replacement if needed."""
    if count < 1:
        raise TrafficError(f"count must be at least 1, got {count}")
    if length < 1:
        raise TrafficError("the selected split has no rows")
    generator = np.random.Generator(np.random.PCG64(seed))
    indices = generator.choice(length, size=count, replace=count > length)
    return [int(index) for index in indices]


def _post(client: Any, url: str, payload: dict[str, Any]) -> Any:
    try:
        response = client.post(url, json=payload, timeout=DEFAULT_TIMEOUT_SECONDS)
    except Exception as exc:
        raise TrafficError(f"request to {url} failed: {exc}") from exc
    if response.status_code != 200:
        detail = response.text[:500]
        raise TrafficError(f"POST {url} returned {response.status_code}: {detail}")
    return response.json()


def simulate_traffic(
    *,
    dataset_dir: Path,
    count: int,
    api_url: str,
    split: TrafficSplit = "validation",
    with_labels: bool = False,
    seed: int = 42,
    client: Any | None = None,
) -> SimulationSummary:
    """Replay ``count`` dataset rows through ``POST /predict`` and ``/labels``."""
    dataset = load_prepared_dataset(dataset_dir)
    frame = split_frame(dataset, split)
    indices = deterministic_indices(len(frame), count, seed)
    base = api_url.rstrip("/")
    versions: Counter[str] = Counter()
    sent = 0
    labeled = 0
    skipped = 0

    owns_client = client is None
    http = client or httpx.Client(timeout=DEFAULT_TIMEOUT_SECONDS)
    try:
        for index in indices:
            row = frame.iloc[index]
            body = _post(http, f"{base}/predict", _payload(dataset, row))
            sent += 1
            version = str(body.get("model_version"))
            versions[version] += 1
            if not with_labels:
                continue
            event_id = body.get("event_id")
            if event_id is None:
                skipped += 1
                continue
            label = int(row["target"])
            _post(http, f"{base}/labels", {"event_id": int(event_id), "label": label})
            labeled += 1
    finally:
        if owns_client:
            http.close()

    return SimulationSummary(
        sent=sent,
        labeled=labeled,
        model_versions=dict(sorted(versions.items())),
        labels_skipped=skipped,
    )


def describe_summary(summary: SimulationSummary) -> list[str]:
    versions = (
        ", ".join(f"{version}={count}" for version, count in summary.model_versions.items())
        or "<none>"
    )
    lines = [
        f"sent: {summary.sent}",
        f"labeled: {summary.labeled}",
        f"model_versions: {versions}",
    ]
    if summary.labels_skipped:
        lines.append(
            f"labels_skipped: {summary.labels_skipped} (predictions whose event write failed)"
        )
    return lines
