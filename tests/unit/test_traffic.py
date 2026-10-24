"""Deterministic traffic simulation through a fake inference client."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from modelops_lab.data import LoadedDataset
from modelops_lab.traffic import deterministic_indices, simulate_traffic


@dataclass
class FakeResponse:
    status_code: int
    payload: dict[str, Any]

    @property
    def text(self) -> str:
        return str(self.payload)

    def json(self) -> dict[str, Any]:
        return self.payload


@dataclass
class FakeClient:
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    next_event_id: int = 100

    def post(self, url: str, json: dict[str, Any], timeout: float | None = None) -> FakeResponse:
        self.calls.append((url, json))
        if url.endswith("/predict"):
            self.next_event_id += 1
            return FakeResponse(
                200,
                {
                    "event_id": self.next_event_id,
                    "request_id": f"req-{self.next_event_id}",
                    "model_version": "1",
                    "prediction": 1,
                    "probability": 0.8,
                },
            )
        return FakeResponse(200, {"event_id": json["event_id"], "label": json["label"]})

    def close(self) -> None:
        pass


def test_simulator_sends_n_predictions_and_n_labels_deterministically(
    loaded_dataset: LoadedDataset, prepared_dataset: Path
) -> None:
    first = FakeClient()
    summary = simulate_traffic(
        dataset_dir=prepared_dataset,
        count=12,
        api_url="http://api.test/",
        split="validation",
        with_labels=True,
        seed=3,
        client=first,
    )

    assert summary.sent == 12
    assert summary.labeled == 12
    assert summary.model_versions == {"1": 12}
    predict_calls = [payload for url, payload in first.calls if url.endswith("/predict")]
    label_calls = [payload for url, payload in first.calls if url.endswith("/labels")]
    assert len(predict_calls) == 12
    assert len(label_calls) == 12
    assert all(set(call) == {"event_id", "label"} for call in label_calls)

    second = FakeClient()
    simulate_traffic(
        dataset_dir=prepared_dataset,
        count=12,
        api_url="http://api.test",
        split="validation",
        with_labels=True,
        seed=3,
        client=second,
    )
    second_predict = [payload for url, payload in second.calls if url.endswith("/predict")]
    assert predict_calls == second_predict


def test_simulator_skips_labels_when_the_event_write_failed(
    loaded_dataset: LoadedDataset, prepared_dataset: Path
) -> None:
    class NullEventClient(FakeClient):
        def post(
            self, url: str, json: dict[str, Any], timeout: float | None = None
        ) -> FakeResponse:
            if url.endswith("/predict"):
                self.calls.append((url, json))
                return FakeResponse(
                    200,
                    {
                        "event_id": None,
                        "request_id": "req",
                        "model_version": "1",
                        "prediction": 0,
                        "probability": 0.2,
                    },
                )
            return super().post(url, json, timeout)

    client = NullEventClient()
    summary = simulate_traffic(
        dataset_dir=prepared_dataset,
        count=3,
        api_url="http://api.test",
        with_labels=True,
        seed=1,
        client=client,
    )

    assert summary.sent == 3
    assert summary.labeled == 0
    assert summary.labels_skipped == 3


def test_deterministic_indices_repeat_and_handle_replacement() -> None:
    assert deterministic_indices(10, 4, 5) == deterministic_indices(10, 4, 5)
    replaced = deterministic_indices(3, 7, 5)
    assert len(replaced) == 7
    assert set(replaced) <= {0, 1, 2}
