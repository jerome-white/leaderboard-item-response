import sys
import csv
import ast
from argparse import ArgumentParser
from dataclasses import fields, asdict
from urllib.parse import urlparse, urlunparse

import requests

from mylib import Logger, AuthorModel

def nocomment(word):
    index = word.find('#')
    return word if index < 0 else word[:index]

def get(url, flagged):
    inside = False
    response = requests.get(urlunparse(url))

    for i in response.iter_lines():
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

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--source', type=urlparse)
    arguments.add_argument('--variable-name', default='FLAGGED')
    args = arguments.parse_args()

    fieldnames = [ x.name for x in fields(AuthorModel) ]
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)

    response = ''.join(get(args.source, args.variable_name))
    for i in ast.literal_eval(response):
        try:
            (author, model) = (x.strip() for x in i.split('/'))
        except ValueError:
            Logger.error(f'Bad author/model: {i}')
            continue
        am = AuthorModel(author, model)
        writer.writerow(asdict(am))
