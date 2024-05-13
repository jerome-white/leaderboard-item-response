import string
import functools as ft
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

from .date_utils import hf_datetime

@ft.cache
def _clean(name, delimiter='_'):
    letters = []
    for n in name:
        l = delimiter if n in string.punctuation else n
        letters.append(l)

    return ''.join(letters)

@dataclass
class CreationDate:
    date: datetime

    def __post_init__(self):
        self.date = hf_datetime(self.date)

@dataclass
class EvaluationTask:
    task: str
    category: str

    def __init__(self, info):
        (_, name, _) = info.split('|')
        parts = name.split('_', maxsplit=1)
        n = len(parts)
        assert 0 < n <= 2

        self.task = parts[0]
        self.category = parts[1] if n > 1 else ''

    def to_path(self):
        args = map(_clean, (self.task, self.category))
        return Path(*args)

@dataclass
class AuthorModel:
    author: str
    model: str

    def __init__(self, name):
        # parse the values
        (lhs, rhs) = map(name.find, ('_', '__'))
        if lhs < 0:
            raise ValueError(f'Cannot parse name {name}')
        (_lhs, _rhs) = (lhs, rhs)

        # calculate the bounds
        if lhs == rhs:
            rhs += 2
        elif rhs < 0:
            lhs += 1
        else:
            lhs += 1
            rhs += 2

        # extract the names
        if lhs == _lhs:
            self.author = None
        elif rhs < 0:
            self.author = name[lhs:]
        else:
            self.author = name[lhs:_rhs]
        self.model = None if rhs == _rhs else name[rhs:]
