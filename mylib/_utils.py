import random
import functools as ft
from pathlib import Path
from dataclasses import dataclass, field, astuple
from urllib.parse import ParseResult, urlunparse

@dataclass(frozen=True)
class SubmissionInfo:
    benchmark: str
    subject: str
    author: str
    model: str

    def to_path(self, suffix=None):
        path = Path(*astuple(self))
        if suffix is not None:
            path = path.with_suffix(suffix)
        return path

    @classmethod
    def from_path(cls, path):
        (model, _) = path.stem.split('.', maxsplit=1)
        return cls(*path.parent.parts, model)

@dataclass
class Experiment:
    benchmark: str
    name: str
    subjects: list = field(default_factory=list)

    def __iter__(self):
        yield from self.subjects

class DatasetPathHandler:
    _netloc = 'datasets'

    def __init__(self):
        kwargs = {
            'scheme': 'hf',
            'netloc': self._netloc,
        }
        for i in ParseResult._fields:
            kwargs.setdefault(i, None)
        self.url = ParseResult(**kwargs)

    @ft.singledispatchmethod
    def to_url(self, path):
        raise TypeError(type(path))

    @to_url.register
    def _(self, path: Path):
        try:
            path = path.relative_to(self._netloc)
        except ValueError:
            pass

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
