import sys
import csv
from argparse import ArgumentParser

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--parameter', action='append')
    args = arguments.parse_args()

    parameters = set(args.parameter)

    reader = csv.DictReader(sys.stdin)
    writer = csv.DictWriter(sys.stdout, fieldnames=reader.fieldnames)
    writer.writeheader()

    for row in reader:
        if row['parameter'] in parameters:
            writer.writerow(row)
