import sys
import csv
from argparse import ArgumentParser

# Reproduces build-ids.py's DocumentHandler numbering (first-seen order,
# 1-indexed) so a --exclude list of document_id/q_i values - computed
# against the already-built stan.json/diagnostics.csv - can be applied
# here, before build-ids.py ever runs. Dropping at this pre-ID stage
# means build-ids.py renumbers what's left contiguously on its own;
# nothing downstream needs patched index arithmetic.
class DocumentHandler:
    def __init__(self):
        self.db = {}

    def __call__(self, data):
        return self.db.setdefault(data['document'], len(self.db) + 1)

def scanf(fp, exclude: set):
    handle = DocumentHandler()
    reader = csv.DictReader(fp)
    writer = None

    for row in reader:
        document_id = handle(row)
        if document_id not in exclude:
            if writer is None:
                writer = csv.DictWriter(sys.stdout, fieldnames=reader.fieldnames)
                writer.writeheader()
            writer.writerow(row)

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--exclude', type=int, action='append', required=True)
    args = arguments.parse_args()

    scanf(sys.stdin, set(args.exclude))
