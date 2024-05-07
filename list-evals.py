import os
from argparse import ArgumentParser

from huggingface_hub import HfApi

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--author', default='open-llm-leaderboard')
    args = arguments.parse_args()

    api = HfApi(token=os.getenv('HF_BEARER_TOKEN'))
    for i in api.list_datasets(author=args.author, search='details_'):
        print(i.id)
