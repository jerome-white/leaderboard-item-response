import sys
import csv
from pathlib import Path
from argparse import ArgumentParser
from multiprocessing import Pool

from mylib import Logger, FileChecksum

def load(path):
    for i in path.rglob('*.csv.gz'):
        checksum = FileChecksum(i)
        if not checksum:
            Logger.error(i)
            continue

        yield pd.read_csv(i, compression='gzip')

def func(args):
    Logger.info(args)
    return (pd
            .concat(load(args))
            .to_dict(orient='records'))

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--index-path', type=Path)
    arguments.add_argument('--workers', type=int)
    args = arguments.parse_args()

    with Pool(args.workers) as pool:
        writer = None
        for i in pool.imap_unordered(func, args.index_path.iterdir()):
            if i:
                if writer is None:
                    writer = csv.DictWriter(sys.stdout, fieldnames=i[0])
                    writer.writeheader()
                writer.writerows(i)
