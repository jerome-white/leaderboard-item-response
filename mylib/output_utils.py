import csv
from io import TextIOWrapper
from urllib.parse import ParseResult, urlunparse

import pandas as pd
import awswrangler as wr

class ChunkedDataWriter:
    def __init__(self, chunk_size: int):
        self.chunk_size = chunk_size
        self.cache = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.cache:
            self.flush()

    def write(self, data: dict):
        self.cache.append(data)
        if len(self.cache) >= self.chunk_size:
            self.flush()
            self.cache = []

        return not bool(self.cache)

    def flush(self):
        raise NotImplementedError()

class CSVWriter(ChunkedDataWriter):
    def	__init__(self, chunk_size: int, fp: TextIOWrapper):
        super().__init__(chunk_size)

        self.fp = fp
        self.writer = None

    def flush(self):
        if self.writer is None:
            fieldnames = self.cache[0][0]
            self.writer = csv.DictWriter(self.fp, fieldnames=fieldnames)
            self.writer.writeheader()
        self.writer.writerows(self.cache)

class SimpleStorageWriter(ChunkedDataWriter):
    def __init__(self, chunk_size: int, bucket: ParseResult):
        super().__init__(chunk_size)
        self.path = urlunparse(bucket)

    def flush(self):
        df = pd.DataFrame.from_records(self.cache)
        wr.s3.to_parquet(
            df=df,
            path=self.path,
            dataset=True,
            use_threads=True,
            compression='gzip',
            partition_cols=[
                'evaluation',
                'author',
            ],
            dtype={
                'date': 'date',
            },
        )
