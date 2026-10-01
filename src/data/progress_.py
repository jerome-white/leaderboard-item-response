import sys
import json
import logging
import collections as cl
from datetime import datetime
from dataclasses import dataclass, fields

@dataclass
class Tally:
    found: bool = True
    complete: bool = False

class RowHandler:
    _sep = '/'

    def __init__(self, script, level):
        self.script = script
        self.info = '{} {}_.py'.format(logging.getLevelName(level), script)

    def __str__(self):
        return self.script

    def __call__(self, row):
        if row.find(self.info) < 0:
            raise LookupError()
        (_, message) = row.split(']')

        return self.handle(message.strip())

    def handle(self, message):
        raise NotImplementedError()

class ListHandler(RowHandler):
    def __init__(self):
        super().__init__('list', logging.INFO)

    def handle(self, message):
        return message

class DownloadHandler(RowHandler):
    def __init__(self):
        super().__init__('download', logging.INFO)

    def handle(self, message):
        (_, *org, _) = message.split(self._sep, maxsplit=3)
        return self._sep.join(org)

class ReduceHandler(RowHandler):
    def __init__(self):
        super().__init__('reduce', logging.WARNING)

    def handle(self, message):
        (skip, path) = message.split(maxsplit=1)
        if skip == 'skipping':
            name = self.standardize(path)
            parts = (
                'open-llm-leaderboard',
                f'{name}-details',
            )

            return self._sep.join(parts)

    def standardize(self, path):
        (*_, author, model) = path.split(self._sep)
        if author != '_':
            model = f'{author}__{model}'

        return model

if __name__ == '__main__':
    records = cl.defaultdict(Tally)
    handlers = (
        ('found', ListHandler()),
        ('complete', ReduceHandler()),
        ('complete', DownloadHandler()),
    )

    for line in sys.stdin:
        for (n, h) in handlers:
            try:
                model = h(line)
            except LookupError:
                continue

            tally = records[model]
            setattr(tally, n, True)
            break

    (n, N) = (0, 0)
    for v in records.values():
        n += v.complete
        N += v.found
    m = N - n

    report = {
        'date': datetime.now().strftime('%c'),
        'found': N,
        'completed': {
            'total': n,
            'frac': n / N,
        },
        'remaining': {
            'total': m,
            'frac': m / N,
        },
    }

    print(json.dumps(report, indent=3))
