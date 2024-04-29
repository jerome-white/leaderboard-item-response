import csv
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import urlunparse

import pandas as pd
import awswrangler as wr

class ChunkedDataWriter:
    def __init__(self, destination, chunk_size):
        self.chunk_size = chunk_size
        self.destination = destination
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
    def __init__(self, destination, chunk_size):
        super().__init__(urlunparse(destination), chunk_size)

    def flush(self):
        df = (pd
              .DataFrame
              .from_records(self.cache)
              .astype({
                  'date': 'datetime64[ns]',
                  'value': float,
              }))
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
