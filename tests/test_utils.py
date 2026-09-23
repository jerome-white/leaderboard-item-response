import unittest
from pathlib import Path

from mylib import Dataset, SubmissionInfo

class DatasetTestCase(unittest.TestCase):
    def test_from_fullname_splits_namespace_and_name(self):
        self.assertEqual(Dataset.from_fullname('0-hero/Matter-0.2-7B-DPO'),
                          Dataset('0-hero', 'Matter-0.2-7B-DPO'))

    def test_from_fullname_without_separator_raises_value_error(self):
        with self.assertRaises(ValueError):
            Dataset.from_fullname('no-separator-here')

    def test_from_flattened_unflattens_first_double_underscore_only(self):
        self.assertEqual(Dataset.from_flattened('BoltMonkey__Neural__Daredevil-7B'),
                          Dataset('BoltMonkey', 'Neural__Daredevil-7B'))

    def test_from_leaderboard_strips_namespace_prefix_then_unflattens(self):
        fullname = 'open-llm-leaderboard/BoltMonkey__NeuralDaredevil-7B-details'
        self.assertEqual(Dataset.from_leaderboard(fullname, 'open-llm-leaderboard'),
                          Dataset('BoltMonkey', 'NeuralDaredevil-7B-details'))

class SubmissionInfoPathTestCase(unittest.TestCase):
    def test_to_path_appends_suffix_to_dotted_model_name(self):
        info = SubmissionInfo('bbh', 'boolean_expressions', 'upstage', 'SOLAR-10.7B-v1.0')
        self.assertEqual(
            info.to_path('.csv.gz'),
            Path('bbh', 'boolean_expressions', 'upstage', 'SOLAR-10.7B-v1.0.csv.gz'),
        )

    def test_to_path_without_suffix_is_unchanged(self):
        info = SubmissionInfo('bbh', 'boolean_expressions', 'upstage', 'SOLAR-10.7B-v1.0')
        self.assertEqual(
            info.to_path(),
            Path('bbh', 'boolean_expressions', 'upstage', 'SOLAR-10.7B-v1.0'),
        )

    def test_from_path_recovers_dotted_model_name(self):
        path = Path('bbh', 'boolean_expressions', 'upstage', 'SOLAR-10.7B-v1.0.csv.gz')
        info = SubmissionInfo.from_path(path, '.csv.gz')
        self.assertEqual(
            info,
            SubmissionInfo('bbh', 'boolean_expressions', 'upstage', 'SOLAR-10.7B-v1.0'),
        )

    def test_round_trip_for_dotted_model_name(self):
        info = SubmissionInfo('mmlu', 'anatomy', '01-ai', 'Yi-1.5-34B-Chat')
        path = info.to_path('.csv.gz')
        self.assertEqual(SubmissionInfo.from_path(path, '.csv.gz'), info)

    def test_round_trip_for_plain_model_name(self):
        info = SubmissionInfo('gsm8k', '_', 'EleutherAI', 'gpt-j-6b')
        path = info.to_path('.csv.gz')
        self.assertEqual(SubmissionInfo.from_path(path, '.csv.gz'), info)

if __name__ == '__main__':
    unittest.main()
