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
    author: str
    model: str
    subject: str | None

    def __post_init__(self):
        if self.subject is None:
            self.subject = '_'

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
