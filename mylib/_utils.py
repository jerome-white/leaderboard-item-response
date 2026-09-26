import random
import sqlite3
import functools as ft
from typing import ClassVar
from pathlib import Path
from dataclasses import dataclass, field, astuple
from urllib.parse import ParseResult, urlunparse
from collections.abc import Iterable, Iterator

@dataclass
class Document:
    question: str
    label: str | None = None

# Benchmarks whose downstream aggregation needs a per-question category,
# and the key under which that category lives in the raw HF doc.
SUBJECT_KEYS = {
    'mmlu': 'category',
    'gpqa': 'High-level domain',
}

class QuestionBank:
    _table = 'questions'
    _schema = f'''
        CREATE TABLE IF NOT EXISTS {_table} (
            benchmark TEXT NOT NULL,
            subject   TEXT NOT NULL,
            doc_hash  TEXT NOT NULL,
            label     TEXT,
            PRIMARY KEY (benchmark, subject, doc_hash)
        )
    '''

    @classmethod
    def connect(cls, path) -> sqlite3.Connection:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path)
        connection.execute('PRAGMA journal_mode=WAL')
        connection.execute('PRAGMA busy_timeout=5000')
        connection.execute(cls._schema)
        return connection

    def __init__(self, connection: sqlite3.Connection, benchmark: str, subject: str):
        self.connection = connection
        self.benchmark = benchmark
        self.subject = subject

    def __iter__(self) -> Iterator[Document]:
        cursor = self.connection.execute(
            f'SELECT doc_hash, label FROM {self._table} '
            'WHERE benchmark = ? AND subject = ?',
            (self.benchmark, self.subject),
        )
        for (doc_hash, label) in cursor:
            yield Document(doc_hash, label)

    def write(self, documents: Iterable[Document]) -> None:
        rows = (
            (self.benchmark, self.subject, d.question, d.label)
            for d in documents
        )
        self.connection.executemany(
            f'INSERT OR IGNORE INTO {self._table} '
            '(benchmark, subject, doc_hash, label) VALUES (?, ?, ?, ?)',
            rows,
        )
        self.connection.commit()

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
        sep = cls._unknown * 2
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
