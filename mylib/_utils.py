import json
import random
import functools as ft
from typing import ClassVar
from pathlib import Path
from dataclasses import dataclass, field, astuple, asdict
from urllib.parse import ParseResult, urlunparse
from collections.abc import Iterable, Iterator

@dataclass
class Document:
    question: str
    content: dict

@dataclass
class DocumentBank:
    benchmark: str
    subject: str
    documents: list = field(default_factory=list)

    def __iter__(self):
        yield from self.documents

class QuestionBank:
    _suffix = '.jsonl'

    def __init__(self, root, benchmark, subject):
        fname = f'{subject}{self._suffix}'
        self.path = root.joinpath(benchmark, fname)

    def __iter__(self) -> Iterator[Document]:
        with self.path.open() as fp:
            for line in fp:
                doc = json.loads(line)
                yield Document(**doc)

    def printf(self, documents: Iterable[Document]) -> None:
        with self.path.open('a') as fp:
            for d in documents:
                print(json.dumps(asdict(d)), file=fp)

@dataclass(frozen=True)
class Dataset:
    namespace: str
    name: str
    _sep: ClassVar[str] = '/'
    _unknown: ClassVar[str] = '_'

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
        sep = self._unknown * 2
        if sep not in name:
            return cls(cls._unknown, name)
        fullname = name.replace(sep, cls._sep, 1)

        return cls.from_fullname(fullname)

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
