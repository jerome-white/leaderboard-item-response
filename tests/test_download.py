import queue
import unittest
import importlib.util
from pathlib import Path
from unittest.mock import patch, MagicMock

from mylib import Backoff

_path = Path(__file__).resolve().parent.parent / 'src' / 'data' / 'download_.py'
_spec = importlib.util.spec_from_file_location('download_', _path)
download_ = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(download_)

class FakeAggregate:
    def __init__(self):
        self.calls = []

    def __call__(self, dbank):
        self.calls.append(dbank)

class DrainTestCase(unittest.TestCase):
    def test_drains_everything_currently_available(self):
        incoming = queue.Queue()
        incoming.put('a')
        incoming.put(None)
        incoming.put('b')
        aggregate = FakeAggregate()

        n = download_.drain(incoming, aggregate)

        self.assertEqual(n, 3)
        self.assertEqual(aggregate.calls, ['a', 'b'])

    def test_returns_zero_when_nothing_available(self):
        aggregate = FakeAggregate()

        self.assertEqual(download_.drain(queue.Queue(), aggregate), 0)
        self.assertEqual(aggregate.calls, [])

class CollectTestCase(unittest.TestCase):
    def test_drains_incrementally_and_catches_stragglers_at_the_end(self):
        outgoing = queue.Queue()
        incoming = queue.Queue()
        aggregate = FakeAggregate()

        def reader():
            yield 'row1'
            incoming.put('result1')  # ready before the next feed's drain
            yield 'row2'
            yield 'row3'
            incoming.put('result2')  # only ready after feeding is done
            incoming.put('result3')

        download_.collect(reader(), outgoing, incoming, aggregate)

        self.assertEqual(outgoing.qsize(), 3)
        self.assertCountEqual(aggregate.calls, ['result1', 'result2', 'result3'])

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
