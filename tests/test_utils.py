import unittest
import sqlite3
import tempfile
import multiprocessing
from pathlib import Path
from types import SimpleNamespace

from mylib import (
    Dataset,
    DatasetPathHandler,
    Document,
    MetadataBank,
    MetadataBankWorker,
    ModelInfo,
    SubmissionInfo,
    retry_after,
)

def _enter_question_bank(path, barrier):
    barrier.wait()
    with MetadataBankWorker(path):
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

class MetadataBankTestCase(unittest.TestCase):
    def make(self, tmp):
        return MetadataBankWorker(Path(tmp, 'questions.sqlite'))

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
                        "PRAGMA table_info(benchmark_questions)"
                    )
                }
            finally:
                connection.close()

        self.assertIn('doc_id', columns)
        self.assertNotIn('doc_hash', columns)

    def test_put_then_get_round_trips_documents(self):
        info = SubmissionInfo('mmlu', 'u.s._history', 'org', 'model')
        documents = [Document('q1', 'history'), Document('q2', 'history')]

        with tempfile.TemporaryDirectory() as tmp, self.make(tmp) as db:
            db.put_questions(info, documents)
            result = list(db.get_questions(info))

        self.assertEqual(result, documents)

    def test_put_then_get_round_trips_models(self):
        models = [
            ModelInfo('org', 'model-a', 'chat', 'bfloat16', 7.0, False),
            ModelInfo('org', 'model-b', 'merge', 'float16', 13.0, True),
        ]

        with tempfile.TemporaryDirectory() as tmp, self.make(tmp) as db:
            db.put_models(models)
            result = list(db.get_models())

        self.assertCountEqual(result, models)

    def test_put_ignores_a_model_already_present(self):
        model = ModelInfo('org', 'model-a', 'chat', 'bfloat16', 7.0, False)

        with tempfile.TemporaryDirectory() as tmp, self.make(tmp) as db:
            db.put_models([model])
            db.put_models([model])
            result = list(db.get_models())

        self.assertEqual(result, [model])

    def test_put_then_get_round_trips_a_numeric_doc_id_as_an_int(self):
        # doc_id is the sample's positional index in lm-evaluation-
        # harness output - structurally an integer, not an opaque
        # label. A TEXT column would silently coerce it to a string
        # on the way in.
        info = SubmissionInfo('mmlu', 'physics', 'org', 'model')

        with tempfile.TemporaryDirectory() as tmp, self.make(tmp) as db:
            db.put_questions(info, [Document(0, 'physics')])
            result = list(db.get_questions(info))

        self.assertEqual(result, [Document(0, 'physics')])
        self.assertIsInstance(result[0].question, int)

    def test_put_ignores_a_doc_id_already_present(self):
        info = SubmissionInfo('mmlu', 'u.s._history', 'org', 'model')

        with tempfile.TemporaryDirectory() as tmp, self.make(tmp) as db:
            db.put_questions(info, [Document('q1', 'history')])
            db.put_questions(info, [Document('q1', 'history')])
            result = list(db.get_questions(info))

        self.assertEqual(result, [Document('q1', 'history')])

    def test_get_is_scoped_to_its_own_benchmark_and_subject(self):
        info_a = SubmissionInfo('mmlu', 'u.s._history', 'org', 'model')
        info_b = SubmissionInfo('mmlu', 'anatomy', 'org', 'model')
        info_c = SubmissionInfo('gpqa', 'u.s._history', 'org', 'model')

        with tempfile.TemporaryDirectory() as tmp, self.make(tmp) as db:
            db.put_questions(info_a, [Document('q1', 'x')])
            db.put_questions(info_b, [Document('q2', 'y')])
            db.put_questions(info_c, [Document('q3', 'z')])
            result = list(db.get_questions(info_a))

        self.assertEqual(result, [Document('q1', 'x')])

    def test_documents_persist_across_separate_connections(self):
        info = SubmissionInfo('mmlu', 'u.s._history', 'org', 'model')

        with tempfile.TemporaryDirectory() as tmp:
            with self.make(tmp) as db:
                db.put_questions(info, [Document('q1', 'history')])

            with self.make(tmp) as db:
                result = list(db.get_questions(info))

        self.assertEqual(result, [Document('q1', 'history')])

    def test_worker_raises_when_not_initialized(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, 'subdir', 'questions.sqlite')
            with self.assertRaises(FileNotFoundError):
                with MetadataBankWorker(path):
                    pass

    def test_initialize_creates_the_schema_on_its_own(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, 'questions.sqlite')
            MetadataBank(path).initialize()

            connection = sqlite3.connect(path)
            try:
                tables = connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            finally:
                connection.close()

        self.assertIn(('benchmark_questions',), tables)
        self.assertIn(('models',), tables)

    def test_workers_initialized_up_front_do_not_race_to_create_the_schema(self):
        ctx = multiprocessing.get_context('fork')

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, 'questions.sqlite')
            MetadataBank(path).initialize()

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
