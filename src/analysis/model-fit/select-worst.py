import sys
from argparse import ArgumentParser

import pandas as pd

if __name__ == '__main__':
    arguments = ArgumentParser()
    arguments.add_argument('--top-k', type=int)
    args = arguments.parse_args()

    df = (pd
          .read_csv(sys.stdin)
          .sort_values(by=['ess_bulk', 'r_hat'], ascending=[True, False])
          .head(args.top_k))
    df.to_csv(sys.stdout, index=False)
