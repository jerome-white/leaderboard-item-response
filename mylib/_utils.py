import random
import functools as ft
from pathlib import Path
from urllib.parse import ParseResult, urlunparse

class DatasetPathHandler:
    def __init__(self):
        kwargs = {
            'scheme': 'hf',
            'netloc': self.netloc,
        }
        for i in ParseResult._fields:
            kwargs.setdefault(i, None)
        self.url = ParseResult(**kwargs)

    @property
    def netloc(self):
        return 'datasets'

    def strip_netloc(self, path):
        try:
            return path.relative_to(self.netloc)
        except ValueError:
            return path

    def relative_to(self, path: Path) -> Path:
        stripped = self.strip_netloc(path)
        parts = stripped.parts[:2]
        return Path(self.netloc, *parts)

    @ft.singledispatchmethod
    def to_url(self, path):
        raise TypeError(type(path))

    @to_url.register
    def _(self, path: Path):
        path = self.strip_netloc(path)
        return self.url._replace(path=str(path))

    @to_url.register
    def _(self, path: str):
        return self.to_url(Path(path))

    def to_string(self, path):
        return urlunparse(self.to_url(path))

# HF doesn't send Retry-After; it implements the IETF RateLimit draft
# instead, e.g. RateLimit: "api";r=499;t=81 - t is seconds until reset.
def retry_after(err) -> int | None:
    response = getattr(err, 'response', None)
    if response is None:
        return None

    header = response.headers.get('RateLimit')
    if header is None:
        return None

    for field in header.split(';'):
        (key, _, value) = field.strip().partition('=')
        if key == 't':
            try:
                return int(value)
            except ValueError:
                break

class Backoff:
    _backoff_factor = 2

    def __init__(self, backoff, fuzz=0):
        self.backoff = backoff
        self.fuzz = fuzz

    def __iter__(self):
        backoff = self.backoff
        while True:
            if self.fuzz:
                backoff += backoff * random.uniform(-self.fuzz, self.fuzz)
            yield backoff

            backoff *= self._backoff_factor
