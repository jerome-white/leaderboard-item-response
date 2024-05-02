import sys
import csv
import ast
from argparse import ArgumentParser
from dataclasses import asdict
from urllib.parse import urlparse, urlunparse

import requests

from mylib import Logger, EvaluationSet, AuthorModel

def nocomment(word):
    index = word.find('#')
    return word if index < 0 else word[:index]

def retrieve(url):
    response = requests.get(urlunparse(url))
    yield from response.iter_lines()

def extract(response, flagged):
    inside = False

    for i in response:
        line = i.decode()

        if line.lstrip().startswith(flagged):
            inside = True
            index = line.find('{')
            assert index >= 0, f'Bracket expected to follow {flagged}'
            line = line[index:]
        if inside:
            line = nocomment(line)
            if line:
                yield line
        if line.find('}') >= 0:
            break

def gather(data):
    response = ''.join(data)
    for i in ast.literal_eval(response):
        try:
            (author, model) = (x.strip() for x in i.split('/'))
        except ValueError:
            Logger.warning(f'Bad author/model: {i}')
            continue
        yield AuthorModel(author, model)

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--source', type=urlparse)
    arguments.add_argument('--variable-name', default='FLAGGED')
    args = arguments.parse_args()

    flagged = set(gather(extract(retrieve(args.source), args.variable_name)))

    reader = csv.DictReader(sys.stdin)
    writer = csv.DictWriter(sys.stdout, fieldnames=reader.fieldnames)
    writer.writeheader()

    for row in reader:
        ev_set = EvaluationSet(**row)
        auth_mod = ev_set.get_author_model()
        if auth_mod in flagged:
            Logger.error(f'Flagged model: {ev_set}')
            continue
        writer.writerow(asdict(ev_set))
