import sys
import csv
import json
import time
import itertools as it
import functools as ft
import statistics as st
from typing import SupportsFloat
from pathlib import Path
from tempfile import NamedTemporaryFile
from argparse import ArgumentParser
from dataclasses import dataclass, fields, asdict, replace
from urllib.parse import ParseResult, urlunparse
from multiprocessing import Pool, JoinableQueue

import fsspec
import requests
import pandas as pd
from requests import HTTPError
from sqlalchemy.exc import SQLAlchemyError
from huggingface_hub.utils import GatedRepoError, build_hf_headers

from mylib import (
    Backoff,
    DatasetPathHandler,
    Document,
    Logger,
    QuestionBank,
    QuestionBankWorker,
    SubmissionInfo,
    retry_after,
)

#
# Types and functions to evaluation scores. Create new `to_float`s to
# handle special cases.
#
@ft.singledispatch
def to_float(value):
    raise TypeError('{}: {}'.format(type(value), value))

@to_float.register
def _(value: SupportsFloat): # most are float, ifeval.prompt_ is bool
    return float(value)

@to_float.register
def _(value: list): # ifeval.inst_
    return st.fmean(value)

@dataclass
class Result:
    document: str
    metric: str
    score: float

    def __post_init__(self):
        self.score = to_float(self.score)

#
#
#
class DatasetAccessRequestor:
    _url = {
        'scheme': 'https',
        'netloc': 'huggingface.co',
    }
    _endpoint = 'ask-access'

    def __init__(self):
        self.path = DatasetPathHandler()

    def __call__(self, path):
        target = urlunparse(self.to_url(path))
        headers = build_hf_headers()

        response = requests.post(target, headers=headers)
        response.raise_for_status()

    def to_url(self, path):
        body = (self
                .path
                .relative_to(path)
                .joinpath(self._endpoint))

        kwargs = dict(self._url, path=str(body))
        for i in ParseResult._fields:
            kwargs.setdefault(i, None)

        return ParseResult(**kwargs)

@ft.singledispatch
def raise_for_hf_reader_error(err, message):
    raise ConnectionError(message) from err

@raise_for_hf_reader_error.register
def _(err: GatedRepoError | HTTPError, message):
    raise PermissionError(message) from err

class HfFileReader:
    def __init__(self, backoff, retries):
        self.ask = DatasetAccessRequestor()
        self.path = DatasetPathHandler()
        self.backoff = backoff
        self.retries = retries

    def __call__(self, target):
        url = self.path.to_string(target)
        asked = False
        last_err = None

        for delay in it.islice(self.backoff, self.retries):
            try:
                with fsspec.open(url) as fp:
                    for line in fp:
                        yield json.loads(line)
                return
            except GatedRepoError as err:
                last_err = err
                Logger.error(url)
                if not asked:
                    try:
                        self.ask(target)
                    except HTTPError as herr:
                        raise_for_hf_reader_error(herr, target)
                    asked = True
            except Exception as err:
                last_err = err
                Logger.error('%s: %s', type(err).__name__, err)
            delay = retry_after(last_err) or delay
            time.sleep(delay)
        raise_for_hf_reader_error(last_err, target)

class SubmissionReader:
    _metrics = (
        'acc',
        'match',
    )
    _subjects = {
        'mmlu': 'category',
        'gpqa': 'High-level domain',
    }

    def __init__(self, reader, benchmark=None):
        self.reader = reader
        self.subject = self._subjects.get(benchmark)
        self.documents = []

    def __call__(self, submission):
        path = Path(submission['path'])
        for r in self.results(path):
            record = dict(submission)
            record.update(asdict(r))
            yield record

    def results(self, path):
        for line in self.reader(path):
            document = line['doc_id']
            self.store(document, line)
            for (metric, score) in line.items():
                if any(metric.find(x) >= 0 for x in self._metrics):
                    yield Result(document, metric, score)

    def store(self, doc, info):
        label = info['doc'][self.subject] if self.subject else None
        document = Document(doc, label)
        self.documents.append(document)

#
#
#
class AtomicWriter:
    def __init__(self, destination: Path):
        self.destination = destination
        self.source = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.source is not None:
            self.source.replace(self.destination)

    def write(self, df: pd.DataFrame) -> None:
        with NamedTemporaryFile(delete=False) as fp:
            df.to_csv(fp, index=False, compression='gzip')
            self.source = Path(fp.name)

def func(queue: JoinableQueue, args):
    hf_reader = HfFileReader(Backoff(args.backoff, 0.1), args.retries)
    keys = [ x.name for x in fields(SubmissionInfo) ]

    with QuestionBankWorker(args.question_bank) as db:
        while True:
            submission = queue.get()
            Logger.info(submission['path'])

            info = SubmissionInfo(*map(submission.get, keys))
            if not info.subject:
                info = replace(info, subject='_')
            reader = SubmissionReader(hf_reader, submission.get('benchmark'))
            try:
                df = pd.DataFrame.from_records(reader(submission))
                if not df.empty:
                    out = args.output.joinpath(info.to_path('.csv.gz'))
                    out.parent.mkdir(parents=True, exist_ok=True)
                    with AtomicWriter(out) as writer:
                        writer.write(df)
                db.put(info, reader.documents)
            except (PermissionError, ConnectionError, SQLAlchemyError) as err:
                Logger.error('%s: %s', type(err), err)
            finally:
                queue.task_done()

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--output', type=Path)
    arguments.add_argument('--question-bank', type=Path)
    arguments.add_argument('--backoff', type=float, default=2)
    arguments.add_argument('--retries', type=int, default=3)
    arguments.add_argument('--workers', type=int)
    args = arguments.parse_args()

    qbank = QuestionBank(args.question_bank)
    qbank.initialize()

    queue = JoinableQueue()
    initargs = (
        queue,
        args,
    )

    with Pool(args.workers, func, initargs):
        reader = csv.DictReader(sys.stdin)
        for row in reader:
            queue.put(row)
        queue.join()
