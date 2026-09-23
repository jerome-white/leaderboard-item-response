import unittest
from pathlib import Path

from mylib import SubmissionInfo

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
