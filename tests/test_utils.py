import unittest
import sqlite3
import tempfile
import multiprocessing
from pathlib import Path
from types import SimpleNamespace

from mylib import (
    BenchmarkQuestion,
    Dataset,
    DatasetPathHandler,
    ModelDatabase,
    ModelMetadata,
    QuestionDatabase,
    SubmissionInfo,
    retry_after,
)

def _enter_question_bank(path, barrier):
    barrier.wait()
    with QuestionDatabase(path):
        pass

class _FakeHttpError(Exception):
    def __init__(self, headers):
        super().__init__('rate limited')
        self.response = SimpleNamespace(headers=headers)

class RetryAfterTestCase(unittest.TestCase):
    def test_reads_seconds_until_reset_from_ratelimit_header(self):
        err = _FakeHttpError({'RateLimit': '"api";r=499;t=81'})
        self.assertEqual(retry_after(err), 81)

    def test_is_none_without_a_response(self):
        self.assertIsNone(retry_after(Exception('boom')))

    def test_is_none_when_header_is_absent(self):
        err = _FakeHttpError({})
        self.assertIsNone(retry_after(err))

    def test_is_none_when_t_field_is_not_numeric(self):
        err = _FakeHttpError({'RateLimit': '"api";r=499;t=soon'})
        self.assertIsNone(retry_after(err))

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

class QuestionDatabaseTestCase(unittest.TestCase):
    def make(self, tmp):
        return QuestionDatabase(Path(tmp, 'questions.sqlite'))

    def test_schema_keys_rows_by_doc_id_not_doc_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, 'questions.sqlite')
            with self.make(tmp):
                pass

            connection = sqlite3.connect(path)
            try:
                columns = {
                    row[1]
                    for row in connection.execute(
                        "PRAGMA table_info(questions)"
                    )
                }
            finally:
                connection.close()

        self.assertIn('doc_id', columns)
        self.assertNotIn('doc_hash', columns)

    def test_put_then_get_round_trips_documents(self):
        info = SubmissionInfo('mmlu', 'u.s._history', 'org', 'model')
        documents = [
            BenchmarkQuestion('mmlu', 'u.s._history', 'q1', 'history'),
            BenchmarkQuestion('mmlu', 'u.s._history', 'q2', 'history'),
        ]

        with tempfile.TemporaryDirectory() as tmp, self.make(tmp) as db:
            db.put(documents)
            result = list(db.get(info))

        self.assertEqual(result, documents)

    def test_put_then_get_round_trips_a_numeric_doc_id_as_an_int(self):
        # doc_id is the sample's positional index in lm-evaluation-
        # harness output - structurally an integer, not an opaque
        # label. A TEXT column would silently coerce it to a string
        # on the way in.
        info = SubmissionInfo('mmlu', 'physics', 'org', 'model')
        document = BenchmarkQuestion('mmlu', 'physics', 0, 'physics')

        with tempfile.TemporaryDirectory() as tmp, self.make(tmp) as db:
            db.put([document])
            result = list(db.get(info))

        self.assertEqual(result, [document])
        self.assertIsInstance(result[0].doc_id, int)

    def test_put_ignores_a_doc_id_already_present(self):
        info = SubmissionInfo('mmlu', 'u.s._history', 'org', 'model')
        document = BenchmarkQuestion('mmlu', 'u.s._history', 'q1', 'history')

        with tempfile.TemporaryDirectory() as tmp, self.make(tmp) as db:
            db.put([document])
            db.put([document])
            result = list(db.get(info))

        self.assertEqual(result, [document])

    def test_get_is_scoped_to_its_own_benchmark_and_subject(self):
        info_a = SubmissionInfo('mmlu', 'u.s._history', 'org', 'model')
        doc_a = BenchmarkQuestion('mmlu', 'u.s._history', 'q1', 'x')
        doc_b = BenchmarkQuestion('mmlu', 'anatomy', 'q2', 'y')
        doc_c = BenchmarkQuestion('gpqa', 'u.s._history', 'q3', 'z')

        with tempfile.TemporaryDirectory() as tmp, self.make(tmp) as db:
            db.put([doc_a, doc_b, doc_c])
            result = list(db.get(info_a))

        self.assertEqual(result, [doc_a])

    def test_documents_persist_across_separate_connections(self):
        info = SubmissionInfo('mmlu', 'u.s._history', 'org', 'model')
        document = BenchmarkQuestion('mmlu', 'u.s._history', 'q1', 'history')

        with tempfile.TemporaryDirectory() as tmp:
            with self.make(tmp) as db:
                db.put([document])

            with self.make(tmp) as db:
                result = list(db.get(info))

        self.assertEqual(result, [document])

    def test_worker_raises_when_not_initialized(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, 'subdir', 'questions.sqlite')
            with self.assertRaises(FileNotFoundError):
                with QuestionDatabase(path):
                    pass

    def test_initialize_creates_the_schema_on_its_own(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, 'questions.sqlite')
            QuestionDatabase(path).initialize()

            connection = sqlite3.connect(path)
            try:
                tables = connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            finally:
                connection.close()

        self.assertIn(('questions',), tables)
        self.assertIn(('models',), tables)

    def test_workers_initialized_up_front_do_not_race_to_create_the_schema(self):
        ctx = multiprocessing.get_context('fork')

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, 'questions.sqlite')
            QuestionDatabase(path).initialize()

            barrier = ctx.Barrier(8)
            processes = [
                ctx.Process(target=_enter_question_bank, args=(path, barrier))
                for _ in range(8)
            ]
            for p in processes:
                p.start()
            for p in processes:
                p.join()

        self.assertTrue(all(p.exitcode == 0 for p in processes))

class ModelDatabaseTestCase(unittest.TestCase):
    def make(self, tmp):
        return ModelDatabase(Path(tmp, 'questions.sqlite'))

    def test_put_then_get_round_trips_models(self):
        models = [
            ModelMetadata('org', 'model-a', 'chat', 'bfloat16', 7.0, False),
            ModelMetadata('org', 'model-b', 'merge', 'float16', 13.0, True),
        ]

        with tempfile.TemporaryDirectory() as tmp, self.make(tmp) as db:
            db.put(models)
            result = list(db.get())

        self.assertCountEqual(result, models)

    def test_put_ignores_a_model_already_present(self):
        model = ModelMetadata('org', 'model-a', 'chat', 'bfloat16', 7.0, False)

        with tempfile.TemporaryDirectory() as tmp, self.make(tmp) as db:
            db.put([model])
            db.put([model])
            result = list(db.get())

        self.assertEqual(result, [model])

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

    def test_from_flattened_without_double_underscore_uses_placeholder_namespace(self):
        self.assertEqual(Dataset.from_flattened('gpt2-details'), Dataset('_', 'gpt2-details'))

    def test_from_leaderboard_without_double_underscore_uses_placeholder_namespace(self):
        fullname = 'open-llm-leaderboard/gpt2-details'
        self.assertEqual(Dataset.from_leaderboard(fullname, 'open-llm-leaderboard'),
                          Dataset('_', 'gpt2-details'))

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
