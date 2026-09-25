import sys
import csv
from typing import ClassVar
from pathlib import Path
from dataclasses import dataclass, asdict

import pandas as pd

from mylib import Dataset, Logger, SubmissionInfo

@dataclass
class Submission:
    path: Path
    date: pd.Timestamp
    _root: ClassVar[tuple] = (
        'samples',
        'leaderboard',
    )
    _sep: ClassVar[str] = '_'

    def __post_init__(self):
        self.path = Path(self.path)
        self.date = pd.to_datetime(self.date)
        self.n = len(self._root)

    def to_sample(self):
        (*_, info, name) = self.path.parts
        dataset = Dataset.from_flattened(info)
        parts = name.split(self._sep)

        root = tuple(parts[:self.n])
        if root != self._root:
            raise ValueError(f'Bad root: {name}')

        (*rest, _timestamp) = parts[self.n:]
        (benchmark, *subject) = rest
        subject = self._sep.join(subject)

        return SubmissionInfo(
            author=dataset.namespace,
            model=dataset.name,
            benchmark=benchmark,
            subject=subject,
        )

def records(fp):
    reader = csv.DictReader(fp)
    for row in reader:
        submission = Submission(**row)
        try:
            sample = submission.to_sample()
        except ValueError:
            Logger.error(submission.path)
            continue

        rec = {}
        for i in (submission, sample):
            rec.update(asdict(i))

        yield rec

if __name__ == '__main__':
    df = pd.DataFrame.from_records(records(sys.stdin))
    df.to_csv(sys.stdout, index=False)
