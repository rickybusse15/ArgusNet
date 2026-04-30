from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from typing import Generic, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


@dataclass(frozen=True)
class CacheMetrics:
    owner: str
    name: str
    capacity: int
    size: int
    hits: int
    misses: int
    evictions: int

    def to_dict(self) -> dict[str, int | str]:
        return {
            "owner": self.owner,
            "name": self.name,
            "capacity": self.capacity,
            "size": self.size,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
        }


class BoundedLRU(Generic[K, V]):
    """Small bounded LRU cache with explicit ownership and observable counters.

    The cache is deliberately simple: owners define stable hashable keys, values
    are stored as-is, and eviction removes the least-recently-used entry once
    capacity is exceeded.
    """

    def __init__(self, *, owner: str, name: str, capacity: int = 8192) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be greater than zero.")
        self.owner = owner
        self.name = name
        self.capacity = int(capacity)
        self._values: OrderedDict[K, V] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key: K) -> V | None:
        try:
            value = self._values.pop(key)
        except KeyError:
            self.misses += 1
            return None
        self.hits += 1
        self._values[key] = value
        return value

    def put(self, key: K, value: V) -> V:
        if key in self._values:
            self._values.pop(key)
        self._values[key] = value
        while len(self._values) > self.capacity:
            self._values.popitem(last=False)
            self.evictions += 1
        return value

    def get_or_create(self, key: K, factory: Callable[[], V]) -> V:
        cached = self.get(key)
        if cached is not None:
            return cached
        return self.put(key, factory())

    def clear(self) -> None:
        self._values.clear()

    def metrics(self) -> CacheMetrics:
        return CacheMetrics(
            owner=self.owner,
            name=self.name,
            capacity=self.capacity,
            size=len(self._values),
            hits=self.hits,
            misses=self.misses,
            evictions=self.evictions,
        )

    def metrics_dict(self) -> dict[str, int | str]:
        return self.metrics().to_dict()


def cache_metrics_snapshot(caches: Mapping[str, BoundedLRU[object, object]]) -> dict[str, dict]:
    """Return a JSON-friendly metrics snapshot for a named cache collection."""
    return {name: cache.metrics_dict() for name, cache in caches.items()}


def quantized_key(*values: float, cell_m: float) -> tuple[int, ...]:
    """Quantize metre-space values into stable integer cache-key coordinates."""
    cell = max(float(cell_m), 1.0e-9)
    return tuple(int(round(float(value) / cell)) for value in values)

