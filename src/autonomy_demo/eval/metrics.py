from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class LatencyAccumulator:
    samples: dict[str, list[float]] = field(default_factory=dict)

    def record(self, module_name: str, duration_ms: float) -> None:
        if module_name not in self.samples:
            self.samples[module_name] = []
        self.samples[module_name].append(duration_ms)

    def mean(self) -> dict[str, float]:
        return {
            name: (sum(values) / len(values) if values else 0.0)
            for name, values in self.samples.items()
        }

    def percentile(self, p: float) -> dict[str, float]:
        import numpy as np

        return {
            name: float(np.percentile(values, p)) if values else 0.0
            for name, values in self.samples.items()
        }

    def latest(self) -> dict[str, float]:
        return {
            name: values[-1] if values else 0.0
            for name, values in self.samples.items()
        }

