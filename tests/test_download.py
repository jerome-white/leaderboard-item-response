import json
import queue
import unittest
import tempfile
import importlib.util
import pandas as pd
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from mylib import Backoff, Document, MetadataBankWorker, SubmissionInfo

_path = Path(__file__).resolve().parent.parent / 'src' / 'data' / 'download_.py'
_spec = importlib.util.spec_from_file_location('download_', _path)
download_ = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(download_)

class _FakeHttpError(Exception):
    def __init__(self, headers):
        super().__init__('rate limited')
        self.response = SimpleNamespace(headers=headers)

class SubmissionReaderTestCase(unittest.TestCase):
    def test_store_extracts_the_label_for_a_known_benchmark(self):
        reader = download_.SubmissionReader(lambda path: iter([]), benchmark='mmlu')

        reader.store('q1', {'doc': {'category': 'algebra'}})

        self.assertEqual(reader.documents, [Document('q1', 'algebra')])

    def test_store_leaves_the_label_unset_for_an_unmapped_benchmark(self):
        reader = download_.SubmissionReader(lambda path: iter([]), benchmark='bbh')

        reader.store('q1', {'doc': {'category': 'algebra'}})

        self.assertEqual(reader.documents, [Document('q1', None)])

class _BoundedQueue(queue.Queue):
    """A queue.Queue that raises Stop once drained, so func()'s
    infinite while-loop terminates instead of blocking forever."""
    class Stop(Exception):
        pass

    def get(self, *args, **kwargs):
        try:
            return super().get(block=False)
        except queue.Empty:
            raise self.Stop()

class FuncTestCase(unittest.TestCase):
    _submission = {
        'path': 'datasets/org/repo-details/x/samples_mmlu_algebra.json',
        'benchmark': 'mmlu',
        'subject': 'algebra',
        'author': 'org',
        'model': 'x',
    }
    _info = SubmissionInfo('mmlu', 'algebra', 'org', 'x')

    def make_args(self, tmp):
        return SimpleNamespace(
            output=Path(tmp),
            question_bank=Path(tmp, 'questions.sqlite'),
            backoff=0.01,
            retries=1,
        )

    def run_func(self, args):
        tasks = _BoundedQueue()
        tasks.put(self._submission)
        with self.assertRaises(_BoundedQueue.Stop):
            download_.func(tasks, args)

    def documents(self, args):
        with MetadataBankWorker(args.question_bank) as db:
            return list(db.get_questions(self._info))

    def test_writes_results_and_documents_on_success(self):
        rows = [
            {'doc_id': 'q1', 'doc': {'category': 'algebra'}, 'acc': 1.0},
            {'doc_id': 'q2', 'doc': {'category': 'algebra'}, 'acc': 0.0},
        ]
        lines = [json.dumps(r).encode() for r in rows]

        with tempfile.TemporaryDirectory() as tmp:
            args = self.make_args(tmp)

            with patch.object(download_, 'fsspec') as mock_fsspec:
                mock_fsspec.open.return_value = FakeFile(lines)
                self.run_func(args)

            out = Path(tmp, 'mmlu', 'algebra', 'org', 'x.csv.gz')
            self.assertTrue(out.exists())
            documents = self.documents(args)

        self.assertCountEqual(
            documents,
            [Document('q1', 'algebra'), Document('q2', 'algebra')],
        )

    def test_skips_output_and_documents_when_the_reader_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.make_args(tmp)

            with patch.object(download_, 'fsspec') as mock_fsspec:
                mock_fsspec.open.side_effect = Exception('boom')
                self.run_func(args)

            self.assertEqual(list(Path(tmp).rglob('*.csv.gz')), [])
            documents = self.documents(args)

        self.assertEqual(documents, [])

class FakeFile:
    def __init__(self, lines):
        self.lines = lines

    def __enter__(self):
        return iter(self.lines)

    def __exit__(self, *exc):
        return False

class AtomicWriterTestCase(unittest.TestCase):
    def test_writes_the_complete_file_and_leaves_no_temp_file(self):
        df = pd.DataFrame({'a': [1, 2], 'b': ['x', 'y']})

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp, 'result.csv.gz')
            with download_.AtomicWriter(out) as writer:
                writer.write(df)

            self.assertEqual(list(Path(tmp).iterdir()), [out])
            result = pd.read_csv(out, compression='gzip')

        self.assertTrue(result.equals(df))

    def test_leaves_no_partial_file_at_the_final_path_when_the_write_fails(self):
        df = pd.DataFrame({'a': [1, 2]})

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp, 'result.csv.gz')

            with patch.object(pd.DataFrame, 'to_csv', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    with download_.AtomicWriter(out) as writer:
                        writer.write(df)

            self.assertFalse(out.exists())

    def test_temp_file_shares_a_filesystem_with_the_destination(self):
        # The temp file must live alongside the destination, or the
        # final replace() can't be atomic - on POSIX, crossing
        # filesystems raises rather than silently falling back to a
        # copy, so this isn't just a performance nicety.
        df = pd.DataFrame({'a': [1]})

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp, 'result.csv.gz')

            with patch.object(
                    download_,
                    'NamedTemporaryFile',
                    wraps=download_.NamedTemporaryFile,
            ) as mock_ntf:
                with download_.AtomicWriter(out) as writer:
                    writer.write(df)

            (_, kwargs) = mock_ntf.call_args
            self.assertEqual(kwargs.get('dir'), out.parent)

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

    def test_retries_generic_connection_failures_then_raises_connection_error(self):
        reader = self.make(retries=3)

        with patch.object(download_, 'fsspec') as mock_fsspec, \
             patch.object(download_, 'time') as mock_time:
            mock_fsspec.open.side_effect = RuntimeError('peer closed connection')

            with self.assertRaises(ConnectionError):
                list(reader(Path('datasets/org/repo-details/x/file.json')))

        self.assertEqual(mock_fsspec.open.call_count, 3)
        self.assertEqual(mock_time.sleep.call_count, 3)
        reader.ask.assert_not_called()

    def test_succeeds_after_one_generic_connection_failure(self):
        reader = self.make(retries=3)

        with patch.object(download_, 'fsspec') as mock_fsspec, \
             patch.object(download_, 'time') as mock_time:
            mock_fsspec.open.side_effect = [
                RuntimeError('peer closed connection'),
                FakeFile([b'{"a": 1}\n']),
            ]
            result = list(reader(Path('datasets/org/repo-details/x/file.json')))

        self.assertEqual(result, [{'a': 1}])
        reader.ask.assert_not_called()

    def test_sleeps_for_ratelimit_reset_instead_of_own_backoff(self):
        reader = self.make(retries=3)

        with patch.object(download_, 'fsspec') as mock_fsspec, \
             patch.object(download_, 'time') as mock_time:
            mock_fsspec.open.side_effect = [
                _FakeHttpError({'RateLimit': '"api";r=0;t=81'}),
                FakeFile([b'{"a": 1}\n']),
            ]
            result = list(reader(Path('datasets/org/repo-details/x/file.json')))

        self.assertEqual(result, [{'a': 1}])
        mock_time.sleep.assert_called_once_with(81)

    def test_falls_back_to_own_backoff_without_ratelimit_header(self):
        reader = self.make(retries=3)

        with patch.object(download_, 'fsspec') as mock_fsspec, \
             patch.object(download_, 'time') as mock_time:
            mock_fsspec.open.side_effect = [
                RuntimeError('peer closed connection'),
                FakeFile([b'{"a": 1}\n']),
            ]
            result = list(reader(Path('datasets/org/repo-details/x/file.json')))

        self.assertEqual(result, [{'a': 1}])
        mock_time.sleep.assert_called_once()
        (delay,) = mock_time.sleep.call_args.args
        self.assertAlmostEqual(delay, 0.01, delta=0.005)

if __name__ == '__main__':
    unittest.main()
