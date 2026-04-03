from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any

from autonomy_demo.interfaces.contracts import EventBus, Subscriber


class InProcessEventBus(EventBus):
    """Simple in-memory bus for tests and the stub runtime."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Subscriber]] = defaultdict(list)
        self._latest: dict[str, Any] = {}

    def publish(self, topic: str, payload: Any) -> None:
        self._latest[topic] = payload
        for callback in self._subscribers.get(topic, []):
            callback(topic, payload)
        for callback in self._subscribers.get("*", []):
            callback(topic, payload)

    def subscribe(self, topic: str, callback: Subscriber) -> None:
        self._subscribers[topic].append(callback)

    def snapshot(self) -> dict[str, Any]:
        return deepcopy(self._latest)

    def snapshot_topics(self, topics: set[str] | list[str] | tuple[str, ...]) -> dict[str, Any]:
        selected: dict[str, Any] = {}
        for topic in topics:
            if topic in self._latest:
                selected[topic] = deepcopy(self._latest[topic])
        return selected

    def latest_topics(self, topics: set[str] | list[str] | tuple[str, ...]) -> dict[str, Any]:
        return {topic: self._latest[topic] for topic in topics if topic in self._latest}
