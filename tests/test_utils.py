import unittest
import tempfile
from pathlib import Path

from mylib import Dataset, DatasetPathHandler, Document, QuestionBank, SubmissionInfo

class DatasetPathHandlerTestCase(unittest.TestCase):
    def test_strip_netloc_removes_prefix_when_present(self):
        handler = DatasetPathHandler()
        path = Path('datasets', 'open-llm-leaderboard', 'contents')
        self.assertEqual(
            handler.strip_netloc(path),
            Path('open-llm-leaderboard', 'contents'),
        )

    def test_strip_netloc_is_noop_when_prefix_absent(self):
        handler = DatasetPathHandler()
        path = Path('open-llm-leaderboard', 'contents')
        self.assertEqual(handler.strip_netloc(path), path)

    def test_relative_to_extracts_repo_path(self):
        handler = DatasetPathHandler()
        path = Path(
            'datasets', 'open-llm-leaderboard', 'BoltMonkey__Neural-details',
            'BoltMonkey__Neural', 'samples_leaderboard_bbh_2024.json',
        )
        self.assertEqual(
            handler.relative_to(path),
            Path('datasets', 'open-llm-leaderboard', 'BoltMonkey__Neural-details'),
        )

    def test_relative_to_requires_a_path_not_a_string(self):
        handler = DatasetPathHandler()
        with self.assertRaises(AttributeError):
            handler.relative_to('datasets/open-llm-leaderboard/contents')

class QuestionBankTestCase(unittest.TestCase):
    def test_path_appends_suffix_to_dotted_subject(self):
        qbank = QuestionBank(Path('questions'), 'mmlu', 'u.s._history')
        self.assertEqual(qbank.path, Path('questions', 'mmlu', 'u.s._history.jsonl'))

    def test_path_appends_suffix_to_plain_subject(self):
        qbank = QuestionBank(Path('questions'), 'bbh', '_')
        self.assertEqual(qbank.path, Path('questions', 'bbh', '_.jsonl'))

    def test_printf_then_iter_round_trips_documents(self):
        documents = [
            Document('q1', {'doc': 'a'}),
            Document('q2', {'doc': 'b'}),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            qbank = QuestionBank(Path(tmp), 'mmlu', 'u.s._history')
            qbank.path.parent.mkdir(parents=True, exist_ok=True)
            qbank.printf(documents)
            self.assertEqual(list(qbank), documents)

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
