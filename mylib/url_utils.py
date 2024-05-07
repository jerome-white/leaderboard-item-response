import random
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

    def to_url(self, path):
        try:
            path = path.relative_to(self._netloc)
        except ValueError:
            pass

        return self.url._replace(path=str(path))

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
