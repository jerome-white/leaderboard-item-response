import unittest
import tempfile
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from mylib import Backoff, Document, QuestionBank

_path = Path(__file__).resolve().parent.parent / 'src' / 'data' / 'download_.py'
_spec = importlib.util.spec_from_file_location('download_', _path)
download_ = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(download_)

class SubmissionReaderTestCase(unittest.TestCase):
    def test_store_extracts_the_label_for_a_known_benchmark(self):
        reader = download_.SubmissionReader(lambda path: iter([]), benchmark='mmlu')

        reader.store('q1', {'doc': {'category': 'algebra'}})

        self.assertEqual(reader.documents, [Document('q1', 'algebra')])

    def test_store_leaves_the_label_unset_for_an_unmapped_benchmark(self):
        reader = download_.SubmissionReader(lambda path: iter([]), benchmark='bbh')

        reader.store('q1', {'doc': {'category': 'algebra'}})

        self.assertEqual(reader.documents, [Document('q1', None)])

class ProcessTestCase(unittest.TestCase):
    _submission = {
        'path': 'datasets/org/repo-details/x/samples_mmlu_algebra.json',
        'benchmark': 'mmlu',
        'subject': 'algebra',
        'author': 'org',
        'model': 'x',
    }
    _keys = ('benchmark', 'subject', 'author', 'model')

    @staticmethod
    def hf_reader(rows):
        def reader(path):
            yield from rows
        return reader

    def test_writes_results_and_documents_on_success(self):
        rows = [
            {'doc_hash': 'q1', 'doc': {'category': 'algebra'}, 'acc': 1.0},
            {'doc_hash': 'q2', 'doc': {'category': 'algebra'}, 'acc': 0.0},
        ]
        connection = QuestionBank.connect(Path(':memory:'))
        self.addCleanup(connection.close)

        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(output=Path(tmp))
            download_.process(
                self._submission,
                self.hf_reader(rows),
                self._keys,
                connection,
                args,
            )
            out = Path(tmp, 'mmlu', 'algebra', 'org', 'x.csv.gz')
            self.assertTrue(out.exists())

        qbank = QuestionBank(connection, 'mmlu', 'algebra')
        self.assertCountEqual(
            list(qbank),
            [Document('q1', 'algebra'), Document('q2', 'algebra')],
        )

    def test_skips_output_and_documents_when_the_reader_fails(self):
        connection = QuestionBank.connect(Path(':memory:'))
        self.addCleanup(connection.close)

        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(output=Path(tmp))
            download_.process(
                self._submission,
                self.hf_reader_raising(ConnectionError('boom')),
                self._keys,
                connection,
                args,
            )
            self.assertEqual(list(Path(tmp).rglob('*.csv.gz')), [])

        qbank = QuestionBank(connection, 'mmlu', 'algebra')
        self.assertEqual(list(qbank), [])

    @staticmethod
    def hf_reader_raising(err):
        def reader(path):
            raise err
        return reader

class FakeFile:
    def __init__(self, lines):
        self.lines = lines

    def __enter__(self):
        return iter(self.lines)

    def __exit__(self, *exc):
        return False

class HfFileReaderTestCase(unittest.TestCase):
    def make(self, retries=3):
        reader = download_.HfFileReader(Backoff(0.01), retries)
        reader.ask = MagicMock()
        return reader

    def test_retries_a_bounded_number_of_times_then_gives_up(self):
        reader = self.make(retries=3)

        with patch.object(download_, 'fsspec') as mock_fsspec, \
             patch.object(download_, 'time') as mock_time:
            mock_fsspec.open.side_effect = download_.GatedRepoError('gated')

            with self.assertRaises(PermissionError):
                list(reader(Path('datasets/org/repo-details/x/file.json')))

        self.assertEqual(reader.ask.call_count, 1)
        self.assertEqual(mock_fsspec.open.call_count, 3)
        self.assertEqual(mock_time.sleep.call_count, 3)

    def test_succeeds_after_one_retry(self):
        reader = self.make(retries=3)

        with patch.object(download_, 'fsspec') as mock_fsspec, \
             patch.object(download_, 'time') as mock_time:
            mock_fsspec.open.side_effect = [
                download_.GatedRepoError('gated'),
                FakeFile([b'{"a": 1}\n']),
            ]
            result = list(reader(Path('datasets/org/repo-details/x/file.json')))

        self.assertEqual(result, [{'a': 1}])
        self.assertEqual(reader.ask.call_count, 1)

if __name__ == '__main__':
    unittest.main()
