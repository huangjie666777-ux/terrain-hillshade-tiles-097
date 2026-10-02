"""Bounded LRU cache for rendered tiles."""
from __future__ import annotations

import threading
from collections import OrderedDict


class TileCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._lock = threading.Lock()
        self._items: OrderedDict[tuple, bytes] = OrderedDict()

    def get(self, key: tuple) -> bytes | None:
        with self._lock:
            if key not in self._items:
                return None
            self._items.move_to_end(key)
            return self._items[key]

    def put(self, key: tuple, value: bytes) -> None:
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)
