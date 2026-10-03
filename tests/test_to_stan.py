import json
import unittest
import importlib.util
import numpy as np
import pandas as pd
from pathlib import Path

_path = Path(__file__).resolve().parent.parent / 'src' / 'model' / 'to-stan.py'
_spec = importlib.util.spec_from_file_location('to_stan', _path)
to_stan = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(to_stan)

class MyEncoderTestCase(unittest.TestCase):
    def test_encodes_a_pandas_series_as_a_list(self):
        series = pd.Series([1, 2, 3])
        self.assertEqual(
            json.dumps(series, cls=to_stan.MyEncoder),
            json.dumps([1, 2, 3]),
        )

    def test_encodes_a_numpy_integer_scalar(self):
        # Series.max()/.min() - used for 'I' and 'J' - return a
        # numpy scalar, not a plain Python int. (np.float64 is a
        # genuine float subclass so json already handles it without
        # help; np.int64 is not a plain int subclass, so it isn't.)
        self.assertEqual(json.dumps(np.int64(42), cls=to_stan.MyEncoder), '42')

if __name__ == '__main__':
    unittest.main()
