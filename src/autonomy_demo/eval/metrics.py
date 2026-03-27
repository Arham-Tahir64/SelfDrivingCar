from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class LatencyAccumulator:
    samples: dict[str, list[float]]

    def mean(self) -> dict[str, float]:
        return {
            name: (sum(values) / len(values) if values else 0.0)
            for name, values in self.samples.items()
        }

