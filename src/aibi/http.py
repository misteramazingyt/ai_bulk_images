"""Polite HTTP: identifying User-Agent, per-host rate limits, Retry-After-aware
backoff, robots.txt for non-API page fetches, and a seekable HTTP Range file."""

from __future__ import annotations

import io
import threading
import time
import urllib.robotparser
from urllib.parse import urlparse

import requests

from . import __version__

DEFAULT_UA = f"aibi-research/{__version__} (academic research; contact: set AIBI_CONTACT)"


class PoliteSession:
    def __init__(self, user_agent: str = DEFAULT_UA, min_interval: float = 1.0, max_retries: int = 5,
                 per_host_interval: dict[str, float] | None = None, timeout: float = 30.0):
        self.s = requests.Session()
        self.s.headers["User-Agent"] = user_agent
        self.min_interval = min_interval
        self.per_host = per_host_interval or {}
        self.max_retries = max_retries
        self.timeout = timeout
        self._last: dict[str, float] = {}
        self._robots: dict[str, urllib.robotparser.RobotFileParser] = {}
        self._lock = threading.Lock()

    def _wait(self, host: str) -> None:
        interval = self.per_host.get(host, self.min_interval)
        with self._lock:
            delay = self._last.get(host, 0) + interval - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self._last[host] = time.monotonic()

    def allowed_by_robots(self, url: str) -> bool:
        p = urlparse(url)
        base = f"{p.scheme}://{p.netloc}"
        if base not in self._robots:
            rp = urllib.robotparser.RobotFileParser()
            try:
                r = self.s.get(base + "/robots.txt", timeout=self.timeout)
                rp.parse(r.text.splitlines() if r.ok else [])
            except requests.RequestException:
                rp.parse([])
            self._robots[base] = rp
        return self._robots[base].can_fetch(self.s.headers["User-Agent"], url)

    def request(self, method: str, url: str, *, respect_robots: bool = False, **kw) -> requests.Response:
        if respect_robots and not self.allowed_by_robots(url):
            raise PermissionError(f"robots.txt disallows {url}")
        host = urlparse(url).netloc
        kw.setdefault("timeout", self.timeout)
        backoff = 2.0
        for attempt in range(self.max_retries + 1):
            self._wait(host)
            try:
                r = self.s.request(method, url, **kw)
            except (requests.ConnectionError, requests.Timeout):
                if attempt == self.max_retries:
                    raise
                time.sleep(backoff)
                backoff *= 2
                continue
            if r.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                ra = r.headers.get("Retry-After")
                time.sleep(float(ra) if ra and ra.isdigit() else backoff)
                backoff *= 2
                continue
            return r
        return r  # pragma: no cover

    def get(self, url: str, **kw) -> requests.Response:
        return self.request("GET", url, **kw)

    def post(self, url: str, **kw) -> requests.Response:
        return self.request("POST", url, **kw)


class HTTPRangeFile(io.RawIOBase):
    """Read-only seekable file over HTTP Range requests (lets zipfile read one
    member of a remote multi-hundred-MB archive without downloading it)."""

    def __init__(self, session: PoliteSession, url: str):
        self.session, self.url, self.pos = session, url, 0
        r = session.request("HEAD", url, allow_redirects=True)
        r.raise_for_status()
        self.url = r.url  # resolved CDN URL
        self.size = int(r.headers["Content-Length"])

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = max(0, base + offset)
        return self.pos

    def readinto(self, b) -> int:
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(b), self.size) - 1
        r = self.session.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        r.raise_for_status()
        data = r.content
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)
