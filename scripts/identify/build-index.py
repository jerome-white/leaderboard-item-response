import sys
from pathlib import Path
from argparse import ArgumentParser

from mylib import Logger, hash_it

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--index-path', type=Path)
    args = arguments.parse_args()

    hf_root = Path('datasets', 'open-llm-leaderboard')
    for i in sys.stdin:
        path = Path(i.strip())
        name = hash_it(str(path), 12)

        (*_, root, dot) = path.relative_to(hf_root).parents
        assert dot == Path('.')

        output = args.index_path.joinpath(root, name).with_suffix('.info')
        if output.exists():
            Logger.error('%s exists')
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(i)
