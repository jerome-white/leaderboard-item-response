import json
import functools as ft
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import pandas as pd

class MyEncoder(json.JSONEncoder):
    @ft.singledispatchmethod
    def default(self, o):
        return super().default(o)

    @default.register
    def _(self, o: pd.Series):
        return o.to_list()

    @default.register
    def _(self, o: np.integer):
        return int(o)

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--data-file', type=Path)
    args = arguments.parse_args()

    df = pd.read_csv(args.data_file, memory_map=True)

    score = df['score']
    integers = (score
                .apply(float.is_integer)
                .all())
    if not integers:
        raise TypeError(f'[ {args.data_file} ] Non-integer scores')

    (i, j) = (df[x] for x in ('document_id', 'author_model_id'))
    data = {
        'I': i.max(),           # questions
        'J': j.max(),           # persons
        'N': len(df),           # observations
        'q_i': i,               # question for n
        'p_j': j,               # person for n
        'y': score.astype(int), # correctness for n
    }

    print(json.dumps(data, cls=MyEncoder))
