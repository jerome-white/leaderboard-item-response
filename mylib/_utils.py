import json
import random
import functools as ft
from typing import ClassVar
from pathlib import Path
from dataclasses import dataclass, field, astuple
from urllib.parse import ParseResult, urlunparse

@dataclass
class Document:
    question: str
    content: dict

    @classmethod
    def scanf(cls, path):
        with path.open() as fp:
            for line in fp:
                doc = json.loads(line)
                yield cls(**doc)

@dataclass(frozen=True)
class Dataset:
    namespace: str
    name: str
    _sep: ClassVar[str] = '/'

    def __str__(self):
        return self._sep.join(astuple(self))

    @classmethod
    def from_fullname(cls, fullname):
        names = fullname.split(cls._sep, maxsplit=1)
        if len(names) != 2:
            raise ValueError(fullname)
        return cls(*names)

    @classmethod
    def from_flattened(cls, name):
        return cls.from_fullname(name.replace('__', cls._sep, 1))

    @classmethod
    def from_leaderboard(cls, fullname, author):
        prefix = f'{author}{cls._sep}'
        name = fullname.removeprefix(prefix)
        return cls.from_flattened(name)

@dataclass(frozen=True)
class SubmissionInfo:
    benchmark: str
    subject: str
    author: str
    model: str

    def to_path(self, suffix=None):
        (*parents, model) = astuple(self)
        if suffix is not None:
            model += suffix
        return Path(*parents, model)

    @classmethod
    def from_path(cls, path, suffix):
        model = path.name.removesuffix(suffix)
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
