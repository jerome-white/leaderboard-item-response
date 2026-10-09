import unittest
import importlib.util
import pandas as pd
from pathlib import Path

_path = Path(__file__).resolve().parent.parent / 'src' / 'model' / 'init-values.py'
_spec = importlib.util.spec_from_file_location('init_values', _path)
init_values_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(init_values_module)

class InitValuesTestCase(unittest.TestCase):
    def test_easy_items_get_a_low_beta_and_hard_items_a_high_one(self):
        df = pd.DataFrame({
            'document_id':       [1, 1, 1, 1, 2, 2, 2, 2],
            'author_model_id':   [1, 2, 3, 4, 1, 2, 3, 4],
            'score':             [1, 1, 1, 0, 0, 0, 0, 1],
        })
        values = init_values_module.init_values(df)

        (easy, hard) = values['beta']
        self.assertLess(easy, hard)

    def test_alpha_is_left_at_the_prior_mean(self):
        df = pd.DataFrame({
            'document_id':       [1, 1, 2, 2],
            'author_model_id':   [1, 2, 1, 2],
            'score':             [1, 0, 0, 1],
        })
        values = init_values_module.init_values(df)

        self.assertEqual(values['alpha'], [1.0, 1.0])

    def test_theta_free_excludes_the_two_anchored_persons(self):
        df = pd.DataFrame({
            'document_id':       [1, 1, 1, 1],
            'author_model_id':   [1, 2, 3, 4],
            'score':             [1, 1, 0, 0],
        })
        values = init_values_module.init_values(df)

        self.assertEqual(len(values['theta_free']), 2)

if __name__ == '__main__':
    unittest.main()
