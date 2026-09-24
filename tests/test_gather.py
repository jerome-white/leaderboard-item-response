import unittest
import importlib.util
from pathlib import Path

_path = Path(__file__).resolve().parent.parent / 'src' / 'data' / 'gather_.py'
_spec = importlib.util.spec_from_file_location('gather_', _path)
gather_ = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gather_)

class SubmissionToSampleTestCase(unittest.TestCase):
    def make(self, path):
        return gather_.Submission(path=path, date='2024-06-02')

    def test_model_name_containing_double_underscore(self):
        path = 'x/foo__ba__rmodel/samples_leaderboard_bbh_2024-06-02T21-55-15.json'
        sample = self.make(path).to_sample()
        self.assertEqual(sample.author, 'foo')
        self.assertEqual(sample.model, 'ba__rmodel')
        self.assertEqual(sample.benchmark, 'bbh')
        self.assertEqual(sample.subject, '')

    def test_benchmark_with_subject(self):
        path = 'x/foo__bar/samples_leaderboard_mmlu_anatomy_2024-06-02T21-55-15.json'
        sample = self.make(path).to_sample()
        self.assertEqual(sample.author, 'foo')
        self.assertEqual(sample.model, 'bar')
        self.assertEqual(sample.benchmark, 'mmlu')
        self.assertEqual(sample.subject, 'anatomy')

    def test_canonical_model_with_no_author_uses_placeholder(self):
        path = 'x/gpt2/samples_leaderboard_bbh_2024-06-02T21-55-15.json'
        sample = self.make(path).to_sample()
        self.assertEqual(sample.author, '_')
        self.assertEqual(sample.model, 'gpt2')
        self.assertEqual(sample.benchmark, 'bbh')
        self.assertEqual(sample.subject, '')

    def test_mismatched_root_raises_value_error(self):
        path = 'x/foo__bar/results_2024-06-02T21-55-15.json'
        with self.assertRaises(ValueError):
            self.make(path).to_sample()

if __name__ == '__main__':
    unittest.main()
