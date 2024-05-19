import csv
import functools as ft
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import urlunparse
from collections.abc import Iterable

import pandas as pd
import awswrangler as wr

class ChunkedDataWriter:
    def __init__(self, destination, chunk_size):
        self.destination = destination
        self.chunk_size = chunk_size
        self.cache = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.cache:
            self.flush()

    def write_and_flush(self, data, push):
        push(data)
        cached = len(self.cache)
        if cached >= self.chunk_size:
            self.flush()
            self.cache = []
        else:
            cached = 0

        return cached

    @ft.singledispatchmethod
    def write(self, data):
        raise TypeError(type(data).__name__)

    @write.register
    def _(self, data: dict):
        return self.write_and_flush(data, self.cache.append)

    @write.register
    def _(self, data: Iterable):
        return self.write_and_flush(data, self.cache.extend)

    def flush(self):
        raise NotImplementedError()

class CSVFileWriter(ChunkedDataWriter):
    def __init__(self, destination, chunk_size):
        super().__init__(Path(destination.path), chunk_size)

    def flush(self):
        fieldnames = self.cache[0][0]
        with NamedTemporaryFile(mode='w',
                                suffix='.csv',
                                prefix='',
                                dir=self.destination,
                                delete=False) as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.cache)

class SimpleStorageWriter(ChunkedDataWriter):
    _conversions = {
        'date': pd.to_datetime,
        'value': pd.to_numeric,
    }

    def __init__(self, destination, chunk_size):
        super().__init__(urlunparse(destination), chunk_size)
        self.assign = {
            x: lambda z: y(z[x]) for (x, y) in self._conversions.items()
        }

    def flush(self):
        df = (pd
              .DataFrame
              .from_records(self.cache)
              .assign(**self.assign))
        wr.s3.to_parquet(
            df=df,
            path=self.destination,
            dataset=True,
            use_threads=True,
            compression='gzip',
            partition_cols=[
                'task',
                # 'author',
            ],
        )
