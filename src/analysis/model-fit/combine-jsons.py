import sys
import csv
import json
from pathlib import Path
from argparse import ArgumentParser

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--json', type=Path, action='append')
    args = arguments.parse_args()

    iterable = (json.loads(x.read_text()) for x in args.json)

    row = next(iterable)
    writer = csv.DictWriter(sys.stdout, fieldnames=row)
    writer.writeheader()
    writer.writerow(row)

    writer.writerows(iterable)
