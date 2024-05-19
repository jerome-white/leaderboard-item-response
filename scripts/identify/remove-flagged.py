import sys
import ast
from pathlib import Path
from argparse import ArgumentParser
from urllib.parse import urlparse, urlunparse

import requests

from mylib import Logger

class ResponseParser:
    @staticmethod
    def nocomment(word):
        index = word.find('#')
        return word if index < 0 else word[:index]

    def __init__(self, response, flagged):
        self.response = response
        self.flagged = flagged

    def __iter__(self):
        inside = False

        for i in self.response:
            line = i.decode()

            if line.lstrip().startswith(self.flagged):
                inside = True
                index = line.find('{')
                assert index >= 0, f'Bracket expected to follow {self.flagged}'
                line = line[index:]
            if inside:
                line = self.nocomment(line)
                if line:
                    yield line
            if line.find('}') >= 0:
                break

    def __str__(self):
        return ''.join(self)

    def to_dict(self):
        return ast.literal_eval(str(self))

def extract(source, flagged, prefix):
    response = (requests
                .get(urlunparse(source))
                .iter_lines())
    parser = ResponseParser(response, flagged)

    for i in parser.to_dict().keys():
        value = i.replace('/', '__')
        yield f'{prefix}{value}'

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--source', type=urlparse)
    arguments.add_argument('--variable-name', default='FLAGGED')
    arguments.add_argument('--author', default='open-llm-leaderboard')
    arguments.add_argument('--prefix', default='details_')
    args = arguments.parse_args()

    prefix = Path(args.author, args.prefix)
    flagged = set(extract(args.source, args.variable_name, prefix))

    for i in sys.stdin:
        line = i.strip()
        if line in flagged:
            Logger.error(f'Flagged model: {line}')
            continue
        print(line)
