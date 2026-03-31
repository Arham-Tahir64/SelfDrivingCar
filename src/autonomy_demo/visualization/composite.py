from __future__ import annotations

from typing import Any


class CompositeVisualizationSink:
    """Fan-out sink so we can run multiple visualizers together."""

    def __init__(self, sinks: list[Any]) -> None:
        self._sinks = [sink for sink in sinks if sink is not None]

    def attach(self, event_bus) -> None:
        for sink in self._sinks:
            sink.attach(event_bus)

    def flush(self) -> None:
        for sink in self._sinks:
            sink.flush()

    def update_bundle(self, bundle) -> None:
        for sink in self._sinks:
            if hasattr(sink, "update_bundle"):
                sink.update_bundle(bundle)
