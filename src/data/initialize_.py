from pathlib import Path
from argparse import ArgumentParser

from mylib import LeaderboardDatabase

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--database', type=Path)
    args = arguments.parse_args()

    db = LeaderboardDatabase(args.database)
    db.initialize()
