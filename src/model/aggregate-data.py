import sys
import csv
import json
import itertools as it
from pathlib import Path
from argparse import ArgumentParser
from dataclasses import dataclass, fields, asdict
from multiprocessing import Pool, Queue

import pandas as pd

from mylib import Logger, Experiment, SubmissionInfo

#
#
#
@dataclass
class Record:
    author: str
    model: str
    document: str
    score: float
    subject: str

#
#
#
class BenchmarkHandler:
    _r_fields = [ x.name for x in fields(Record) ]

    def __init__(self, info, documents, metric):
        self.info = info
        self.documents = documents
        self.metric = metric

    def __call__(self, df, subject):
        mask = df['metric'] == self.metric
        view = df[mask]
        if view.empty:
            raise ValueError(f'{self.metric} not in data')

        observations = view.itertuples(index=False)
        for i in self.handle(subject, observations):
            args = (getattr(i, x) for x in self._r_fields)
            yield Record(*args, subject=subject)

    def handle(self, subject, observations):
        raise NotImplementedError()

# Subjects included in directory structure
class DirectoryHandler(BenchmarkHandler):
    def handle(self, subject, observations):
        if self.info.subject == subject:
            yield from observations

class BigBenchHard(DirectoryHandler):
    def __init__(self, info, documents):
        super().__init__(info, documents, 'acc_norm')

class MultistepSoftReasoning(DirectoryHandler):
    def __init__(self, info, documents):
        super().__init__(info, documents, 'acc_norm')

class Math(DirectoryHandler):
    def __init__(self, info, documents):
        super().__init__(info, documents, 'exact_match')

# Subjects included in docs
class IndexedCategoryBenchmark(BenchmarkHandler):
    def __init__(self, info, documents, metric, s_key):
        super().__init__(info, documents, metric)
        docs = json.loads(self.documents.read_text())
        self.subjects = { x: y['doc'][s_key] for (x, y) in docs.items() }

    def handle(self, subject, observations):
        for o in observations:
            if self.subjects[o.document] == subject:
                yield o

class MultitaskUnderstanding(IndexedCategoryBenchmark):
    def __init__(self, info, documents):
        super().__init__(info, documents, 'acc', 'category')

class GraduateLevelGoogleProofQA(IndexedCategoryBenchmark):
    # questions are repeated across subjects!
    def __init__(self, info, documents):
        super().__init__(info, documents, 'acc_norm', 'High-level domain')

# Do not have the concept of subject
class NoSubjectBenchmark(BenchmarkHandler):
    def handle(self, subject, observations):
        yield from observations

class AbstractionReasoningCorpus(NoSubjectBenchmark):
    def __init__(self, info, documents):
        super().__init__(info, documents, 'acc_norm')

class GradeSchoolMath8K(NoSubjectBenchmark):
    def __init__(self, info, documents):
        super().__init__(info, documents, 'exact_match')

class InstructionFollowingEval(NoSubjectBenchmark):
    def __init__(self, info, documents):
        super().__init__(info, documents, 'prompt_level_strict_acc')

#
#
#
def func(incoming, outgoing, args):
    experiment = Experiment(**json.loads(args.experiment.read_text()))
    Handler = {
        'bbh': BigBenchHard,
        'arc': AbstractionReasoningCorpus,
        'math': Math,
        'mmlu': MultitaskUnderstanding,
        'musr': MultistepSoftReasoning,
        'gpqa': GraduateLevelGoogleProofQA,
        'gsm8k': GradeSchoolMath8K,
        'ifeval': InstructionFollowingEval,
    }[experiment.benchmark]

    while True:
        path = incoming.get()
        Logger.info(path)

        df = pd.read_csv(path, compression='gzip', memory_map=True)
        info = SubmissionInfo.from_path(path.relative_to(args.data_root))

        handler = Handler(info, args.question_bank)
        for e in experiment:
            try:
                records = list(handler(df, e))
            except ValueError as err:
                Logger.error('%s %s: %s', path, e, err)
                continue
            outgoing.put(records)
        outgoing.put(None)

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--data-root', type=Path)
    arguments.add_argument('--question-bank', type=Path)
    arguments.add_argument('--experiment', type=Path)
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
        experiments = json.loads(args.experiments.read_text())
        root = args.data_root.joinpath(experiments['benchmark'])
        iterable = it.product(
            args.data_root.rglob('*.csv.gz'),
            experiments['subject']
        )

        jobs = 0
        for i in iterable:
            outgoing.put(i)
            jobs += 1

        fieldnames = [ x.name for x in fields(Record) ]
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()

        while jobs:
            rows = incoming.get()
            if rows is None:
                jobs -= 1
            elif rows:
                writer.writerows(map(asdict, rows))
