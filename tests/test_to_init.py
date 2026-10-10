import sys
import json
import unittest
import tempfile
import importlib.util
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path

_path = Path(__file__).resolve().parent.parent / 'src' / 'model' / 'to-init.py'
_spec = importlib.util.spec_from_file_location('to_init', _path)
to_init = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(to_init)

class ItemExtractorTestCase(unittest.TestCase):
    def test_returns_sorted_clipped_means_as_a_numpy_array(self):
        df = pd.DataFrame({
            'document_id': [2, 2, 1, 1],
            'score':       [1, 1, 0, 1],
        })
        extract = to_init.ItemExtractor(df, epsilon=1e-3)

        result = extract('document_id')

        self.assertIsInstance(result, np.ndarray)
        # document 1: mean 0.5; document 2: mean 1.0, clipped to 1-epsilon.
        np.testing.assert_allclose(result, [0.5, 1 - 1e-3])

    def test_clips_extreme_values_away_from_the_boundary(self):
        df = pd.DataFrame({'document_id': [1, 1], 'score': [0, 0]})
        extract = to_init.ItemExtractor(df, epsilon=1e-3)

        result = extract('document_id')

        np.testing.assert_allclose(result, [1e-3])

    def test_uses_its_own_dataframe_not_a_shared_global(self):
        # Regression test: __call__ previously closed over whatever
        # module-level "df" happened to exist instead of self.df,
        # and only worked by coincidence (__main__'s own "df"
        # shared that name). Two instances with different frames
        # catch that regressing.
        df_a = pd.DataFrame({'document_id': [1, 1], 'score': [1, 1]})
        df_b = pd.DataFrame({'document_id': [1, 1], 'score': [0, 0]})
        extract_a = to_init.ItemExtractor(df_a, epsilon=1e-3)
        extract_b = to_init.ItemExtractor(df_b, epsilon=1e-3)

        np.testing.assert_allclose(extract_a('document_id'), [1 - 1e-3])
        np.testing.assert_allclose(extract_b('document_id'), [1e-3])

class MainTestCase(unittest.TestCase):
    def test_produces_valid_json_with_the_expected_shapes(self):
        # End-to-end, not just unit-level: alpha/beta/theta are
        # scipy/numpy values assembled only inside __main__, and
        # MyEncoder's JSON serialization has broken here before
        # (scipy.special.logit on a pandas Series returns a Series,
        # not an ndarray, and MyEncoder only registered the latter) -
        # a unit test of ItemExtractor alone wouldn't have caught
        # that; only an actual run through json.dumps does.
        with tempfile.TemporaryDirectory() as tmp:
            data_file = Path(tmp, 'data.csv')
            pd.DataFrame({
                'document_id':     [1, 1, 1, 1, 2, 2, 2, 2],
                'author_model_id': [1, 2, 3, 4, 1, 2, 3, 4],
                'score':           [1, 1, 1, 0, 0, 0, 0, 1],
            }).to_csv(data_file, index=False)

            result = subprocess.run(
                [sys.executable, str(_path), '--data-file', str(data_file)],
                capture_output=True, text=True, check=True,
            )
            data = json.loads(result.stdout)

        self.assertEqual(data['alpha'], [1.0, 1.0])

        (easy, hard) = data['beta']
        self.assertLess(easy, hard)

        # 4 persons - 2 anchored (Stan's theta[1]/theta[2]).
        self.assertEqual(len(data['theta_free']), 2)

if __name__ == '__main__':
    unittest.main()
