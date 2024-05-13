import sys
import collections as cl
from pathlib import Path
from argparse import ArgumentParser
from dataclasses import dataclass

from mylib import Logger, AuthorModel, hash_it

@dataclass(frozen=True)
class DataGroup:
    target: Path

    def __repr__(self):
        return hash_it(str(self.target.parent), 16)

def group(fp):
    db = cl.defaultdict(set)
    hf_root = Path('datasets', 'open-llm-leaderboard')

    for i in fp:
        path = Path(i.strip())

        (details, *_) = path.relative_to(hf_root).parts
        am = AuthorModel(details)
        key = DataGroup(Path(am.author, am.model))

        db[key].add(path)

    yield from db.items()

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--index-path', type=Path)
    args = arguments.parse_args()

    for (g, v) in group(sys.stdin):
        output = (args
                  .index_path
                  .joinpath(g.target, repr(g))
                  .with_suffix('.info'))
        if output.exists():
            Logger.error('%s exists', output)
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text('{}\n'.format('\n'.join(v)))
