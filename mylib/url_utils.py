import functools as ft
from pathlib import Path
from urllib.parse import ParseResult, urlunparse

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
