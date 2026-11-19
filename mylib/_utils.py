import random
import functools as ft
from pathlib import Path
from dataclasses import dataclass, fields
from urllib.parse import ParseResult, urlunparse

# HF doesn't send Retry-After; it implements the IETF RateLimit draft
# instead:
#
#   RateLimit: "api";r=499;t=81
#
# where t is seconds until reset.
def retry_after(err) -> int | None:
    response = getattr(err, 'response', None)
    if response is None:
        return

    header = response.headers.get('RateLimit')
    if header is None:
        return

    for field in header.split(';'):
        (key, _, value) = field.strip().partition('=')
        if key == 't':
            try:
                return int(value)
            except ValueError:
                break

#
#
#
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

#
#
#
class VariableHandler:
    def __init__(self, start=1):
        self.start = start
        self.db = {}

    def __call__(self, data):
        index = len(self.db) + self.start
        return self.db.setdefault(self.extract(data), index)

    def extract(self, data):
        raise NotImplementedError()

class AuthorModelHandler(VariableHandler):
    @dataclass(frozen=True)
    class AuthorModel:
        author: str
        model: str

    def __init__(self):
        super().__init__()
        self.keys = tuple(x.name for x in fields(self.AuthorModel))

    def extract(self, data):
        args = map(data.get, self.keys)
        return self.AuthorModel(*args)

class DocumentHandler(VariableHandler):
    def extract(self, data):
        return data['document']
