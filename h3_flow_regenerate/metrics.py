from __future__ import annotations

import json
import tempfile
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class MetricEvent:
    kind: str
    timestamp_ns: int = field(default_factory=time.time_ns)
    fields: dict[str, Any] = field(default_factory=dict)


class H3FlowMetrics:
    api_version = 1

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._events: list[MetricEvent] = []
        self._counters: Counter[str] = Counter()

    def event(self, kind: str, **fields: Any) -> None:
        with self._lock:
            self._events.append(MetricEvent(kind=str(kind), fields=dict(fields)))

    def increment(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[str(name)] += int(amount)

    @property
    def events(self) -> tuple[MetricEvent, ...]:
        with self._lock:
            return tuple(self._events)

    @property
    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counters)

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "counters": self.counters,
            "events": [asdict(event) for event in self.events],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.snapshot(), indent=indent, sort_keys=True, default=str)

    def write_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                handle.write(self.to_json() + "\n")
                temporary = Path(handle.name)
            temporary.replace(target)
            return target
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
