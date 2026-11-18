import sys
import csv
import json
from pathlib import Path

if __name__ == '__main__':
    writer = None
    for line in sys.stdin:
        row = json.loads(line)
        if writer is None:
            writer = csv.DictWriter(sys.stdout, fieldnames=row)
            writer.writeheader()
        writer.writerow(row)
