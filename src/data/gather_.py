import sys
import csv
from typing import ClassVar
from pathlib import Path
from argparse import ArgumentParser
from dataclasses import dataclass, asdict, fields

import pandas as pd

from mylib import Logger, SubmissionInfo

@dataclass
class Submission:
    path: Path
    date: pd.Timestamp
    _root: ClassVar[Path] = Path('samples', 'leaderboard')

    def __post_init__(self):
        self.path = Path(self.path)
        self.date = pd.to_datetime(self.date)

    def to_sample(self):
        (*_, info, name) = self.path.parts
        (author, model) = info.split('__')
        path = Path(*name.split('_'))
        (benchmark, *subject) = (path
                                 .relative_to(self._root)
                                 .parent
                                 .parts)
        subject = '_'.join(subject)

        return SubmissionInfo(author, model, benchmark, subject)

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
    arguments = ArgumentParser()
    arguments.add_argument('--output', type=Path)
    args = arguments.parse_args()

    df = pd.DataFrame.from_records(records(sys.stdin))
    df.to_csv(sys.stdout, index=False)
