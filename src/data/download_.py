import sys
import csv
import json
import time
import itertools as it
import functools as ft
import statistics as st
import collections as cl
from typing import SupportsFloat
from pathlib import Path
from argparse import ArgumentParser
from dataclasses import dataclass, fields, asdict, replace
from urllib.parse import ParseResult, urlunparse
from multiprocessing import Pool, Queue
from collections.abc import Iterator

import fsspec
import requests
import pandas as pd
from requests import HTTPError
from huggingface_hub.utils import GatedRepoError, build_hf_headers

from mylib import (
    Backoff,
    DatasetPathHandler,
    Document,
    DocumentBank,
    Logger,
    QuestionBank,
    SubmissionInfo,
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
class DocumentAggregator:
    def __init__(self, destination):
        self.destination = destination
        self.history = cl.defaultdict(set)

    def __call__(self, dbank: DocumentBank) -> None:
        qbank = QuestionBank(self.destination, dbank.benchmark, dbank.subject)
        history = self.setup_and_load(qbank)
        qbank.printf(self.documents(dbank, history))

    def documents(
            self,
            dbank: DocumentBank,
            history: set,
    ) -> Iterator[Document]:
        for doc in dbank:
            if doc.question not in history:
                yield doc
                history.add(doc.question)

    def setup_and_load(self, qbank: QuestionBank) -> set:
        history = self.history[qbank.path]

        if not qbank.path.exists():
            qbank.path.parent.mkdir(parents=True, exist_ok=True)
        elif not history:
            for doc in qbank:
                history.add(doc.question)

        return history

#
#
#
class DatasetAccessRequestor:
    _url = {
        'scheme': 'https',
        'netloc': 'huggingface.co',
    }
    _endpoint = 'ask-access'

    def __call__(self, path):
        target = urlunparse(self.to_url(path))
        headers = build_hf_headers()

        response = requests.post(target, headers=headers)
        response.raise_for_status()

    def to_url(self, path):
        body = path.parts[:3]
        path = Path(*body, 'ask-access')

        kwargs = dict(self._url, path=str(path))
        for i in ParseResult._fields:
            kwargs.setdefault(i, None)

        return ParseResult(**kwargs)

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
                        raise PermissionError(target) from herr
                    asked = True
                time.sleep(delay)
            except Exception as err:
                raise ConnectionError(target) from err

        raise PermissionError(target) from last_err

class SubmissionReader:
    _document_keys = (
        'doc',
        'doc_id',
    )
    _metrics = (
        'acc',
        'match',
    )

    def __init__(self, reader):
        self.reader = reader
        self.documents = []

    def __call__(self, submission):
        path = Path(submission['path'])
        for r in self.results(path):
            record = dict(submission)
            record.update(asdict(r))
            yield record

    def results(self, path):
        for line in self.reader(path):
            document = line['doc_hash']
            self.store(document, line)
            for (metric, score) in line.items():
                if any(metric.find(x) >= 0 for x in self._metrics):
                    yield Result(document, metric, score)

    def store(self, doc, info):
        content = { x: info[x] for x in self._document_keys }
        document = Document(doc, content)
        self.documents.append(document)

#
#
#
def func(incoming, outgoing, args):
    hf_reader = HfFileReader(Backoff(args.backoff, 0.1), args.retries)
    keys = [ x.name for x in fields(SubmissionInfo) ]

    while True:
        submission = incoming.get()
        Logger.info(submission['path'])

        reader = SubmissionReader(hf_reader)
        try:
            df = pd.DataFrame.from_records(reader(submission))
        except (PermissionError, ConnectionError) as err:
            Logger.critical('%s: %s', type(err), err)
            outgoing.put(None)
            continue

        info = SubmissionInfo(*map(submission.get, keys))
        if not info.subject:
            info = replace(info, subject='_')

        if not df.empty:
            out = args.output.joinpath(info.to_path('.csv.gz'))
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out, index=False, compression='gzip')

        dbank = DocumentBank(info.benchmark, info.subject, reader.documents)
        outgoing.put(dbank)

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--output', type=Path)
    arguments.add_argument('--question-bank', type=Path)
    arguments.add_argument('--backoff', type=float, default=2)
    arguments.add_argument('--retries', type=int, default=3)
    arguments.add_argument('--workers', type=int)
    args = arguments.parse_args()

    incoming = Queue()
    outgoing = Queue()
    initargs = (
        outgoing,
        incoming,
        args,
    )

    with Pool(args.workers, func, initargs):
        jobs = 0
        reader = csv.DictReader(sys.stdin)
        for row in reader:
            outgoing.put(row)
            jobs += 1

        aggregate = DocumentAggregator(args.question_bank)
        for _ in range(jobs):
            dbank = incoming.get()
            if dbank is not None:
                aggregate(dbank)
