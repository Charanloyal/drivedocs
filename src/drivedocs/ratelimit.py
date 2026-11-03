"""In-memory, per-client sliding-window rate limiting for the public demo.

One process serves the app (single uvicorn worker), so an in-process limiter is sufficient; it is not shared across
instances. Limits are per client and per bucket (chat, search, upload, delete). Memory is bounded: at most
``max_clients`` (bucket, client) windows are tracked and the least recently used ones are evicted.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from typing import Callable, Deque, Dict, Mapping, Optional, Tuple

WINDOW_SECONDS = 60.0

# Path prefixes (POST/DELETE) → bucket. Reads (GET) are cheap and not limited.
BUCKETS: Tuple[Tuple[str, str, str], ...] = (
    ("POST", "/chat", "chat"),
    ("POST", "/search/", "search"),
    ("POST", "/documents/upload", "upload"),
    ("DELETE", "/documents/", "delete"),
)


def bucket_for(method: str, path: str) -> Optional[str]:
    for bucket_method, prefix, name in BUCKETS:
        if method == bucket_method and (path == prefix or path.startswith(prefix)):
            return name
    return None


def client_id(headers: Mapping[str, str], peer: Optional[str], trust_proxy_headers: bool) -> str:
    """The client address used as the rate-limit key.

    Proxy headers are only honoured when the app runs behind a trusted proxy (e.g. Render). Edge-set headers that the
    edge overwrites are preferred; otherwise the first X-Forwarded-For entry. A forged header can only evade the limit,
    not throttle someone else.
    """
    if trust_proxy_headers:
        for name in ("true-client-ip", "cf-connecting-ip"):
            value = (headers.get(name) or "").strip()
            if value:
                return value[:64]
        forwarded = (headers.get("x-forwarded-for") or "").split(",")[0].strip()
        if forwarded:
            return forwarded[:64]
    return peer or "unknown"


class RateLimiter:
    """Sliding-window limiter: at most ``limits[bucket]`` requests per client per ``window`` seconds."""

    def __init__(
        self,
        limits: Mapping[str, int],
        window: float = WINDOW_SECONDS,
        max_clients: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.limits = {k: int(v) for k, v in limits.items() if int(v) > 0}
        self.window = window
        self.max_clients = max_clients
        self._clock = clock
        self._hits: "OrderedDict[Tuple[str, str], Deque[float]]" = OrderedDict()
        self._lock = threading.Lock()

    def check(self, bucket: str, client: str) -> Optional[float]:
        """Record a request. Returns None if allowed, else the number of seconds until the next request is allowed."""
        limit = self.limits.get(bucket)
        if not limit:
            return None
        now = self._clock()
        key = (bucket, client)
        with self._lock:
            hits = self._hits.get(key)
            if hits is None:
                hits = deque()
                self._hits[key] = hits
                if len(self._hits) > self.max_clients:
                    self._hits.popitem(last=False)
            else:
                self._hits.move_to_end(key)
            while hits and now - hits[0] >= self.window:
                hits.popleft()
            if len(hits) >= limit:
                return max(0.0, self.window - (now - hits[0]))
            hits.append(now)
            return None

    def snapshot(self) -> Dict[str, int]:
        return dict(self.limits)
