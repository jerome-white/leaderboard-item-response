import unittest
import importlib.util
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch, MagicMock

from mylib import Backoff, Dataset, ModelMetadata

_path = Path(__file__).resolve().parent.parent / 'src' / 'data' / 'list_.py'
_spec = importlib.util.spec_from_file_location('list_', _path)
list_ = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(list_)

class _FakeHttpError(Exception):
    def __init__(self, headers):
        super().__init__('rate limited')
        self.response = SimpleNamespace(headers=headers)

class DatasetFileSystemTestCase(unittest.TestCase):
    def test_constructs_hffilesystem_with_expand_info(self):
        with patch.object(list_, 'HfFileSystem') as mock_cls:
            list_.DatasetFileSystem(backoff=[1])
            mock_cls.assert_called_once_with(expand_info=True)

    def test_ls_sleeps_for_ratelimit_reset_instead_of_own_backoff(self):
        fs = list_.DatasetFileSystem(Backoff(5))
        fs.fs = MagicMock()
        fs.fs.ls.side_effect = [_FakeHttpError({'RateLimit': '"api";r=0;t=81'}), ['ok']]

        with patch.object(list_, 'time') as mock_time:
            result = list(fs.ls('open-llm-leaderboard/foo-details'))

        self.assertEqual(result, ['ok'])
        mock_time.sleep.assert_called_once_with(81)

    def test_ls_falls_back_to_own_backoff_without_ratelimit_header(self):
        fs = list_.DatasetFileSystem(Backoff(5))
        fs.fs = MagicMock()
        fs.fs.ls.side_effect = [Exception('transient'), ['ok']]

        with patch.object(list_, 'time') as mock_time:
            result = list(fs.ls('open-llm-leaderboard/foo-details'))

        self.assertEqual(result, ['ok'])
        mock_time.sleep.assert_called_once()
        (delay,) = mock_time.sleep.call_args.args
        self.assertAlmostEqual(delay, 5, delta=0.5)

class DatasetIteratorTestCase(unittest.TestCase):
    def test_handles_a_contents_row_with_no_author(self):
        # Some leaderboard entries (e.g. "gpt2") have no author at
        # all in the contents dataset's fullname field.
        rows = [{'fullname': 'gpt2'}]

        with patch.object(list_, 'load_dataset', return_value=rows):
            datasets = list_.DatasetIterator('open-llm-leaderboard')

        (listing,) = list(datasets)
        self.assertEqual(listing.dataset, Dataset('_', 'gpt2'))

class FlaggedHandlerTestCase(unittest.TestCase):
    _rows = [
        {'fullname': 'org/model-a', 'Flagged': True},
        {'fullname': 'org/model-b', 'Flagged': False},
    ]

    def make(self):
        with patch.object(list_, 'load_dataset', return_value=self._rows):
            datasets = list_.DatasetIterator('open-llm-leaderboard')

        base = list_.ModelHandler('open-llm-leaderboard')
        return list_.FlaggedHandler(base, datasets)

    def test_rejects_a_flagged_model(self):
        model = SimpleNamespace(id='open-llm-leaderboard/org__model-a-details')
        self.assertIsNone(self.make().handle(model))

    def test_passes_through_an_unflagged_model(self):
        model = SimpleNamespace(id='open-llm-leaderboard/org__model-b-details')
        self.assertIs(self.make().handle(model), model)

class DatabaseHandlerTestCase(unittest.TestCase):
    _rows = [
        {
            'fullname': 'org/model-a',
            'Type': 'chat',
            'Precision': 'bfloat16',
            '#Params (B)': 7.0,
            'Merged': False,
            'Flagged': True,
        },
    ]

    def make(self, db):
        with patch.object(list_, 'load_dataset', return_value=self._rows):
            datasets = list_.DatasetIterator('open-llm-leaderboard')

        base = list_.ModelHandler('open-llm-leaderboard')
        return list_.DatabaseHandler(base, datasets, db)

    def test_stores_metadata_for_a_matched_model_and_returns_it_unchanged(self):
        model = SimpleNamespace(id='open-llm-leaderboard/org__model-a-details')
        db = MagicMock()

        result = self.make(db).handle(model)

        self.assertIs(result, model)
        db.put.assert_called_once_with(
            ModelMetadata('org', 'model-a', 'chat', 'bfloat16', 7.0, False),
        )

    def test_passes_through_an_unmatched_model_without_storing_anything(self):
        model = SimpleNamespace(id='open-llm-leaderboard/other__model-x-details')
        db = MagicMock()

        result = self.make(db).handle(model)

        self.assertIs(result, model)
        db.put.assert_not_called()

class ModelHandlerChainTestCase(unittest.TestCase):
    # fullname/Flagged drive FlaggedHandler, the remaining fields
    # drive DatabaseHandler - both handlers read from the one
    # DatasetIterator built in each test's make().
    _rows = [
        {
            'fullname': 'org/model-a',
            'Type': 'chat',
            'Precision': 'bfloat16',
            '#Params (B)': 7.0,
            'Merged': False,
            'Flagged': True,
        },
        {
            'fullname': 'org/model-b',
            'Type': 'merge',
            'Precision': 'float16',
            '#Params (B)': 13.0,
            'Merged': True,
            'Flagged': False,
        },
    ]

    def make(self, db):
        with patch.object(list_, 'load_dataset', return_value=self._rows):
            datasets = list_.DatasetIterator('open-llm-leaderboard')

        base = list_.ModelHandler('open-llm-leaderboard')
        flagged = list_.FlaggedHandler(base, datasets)
        return list_.DatabaseHandler(flagged, datasets, db)

    def test_flagged_models_are_filtered_from_the_stream_but_still_recorded(self):
        models = [
            SimpleNamespace(id='open-llm-leaderboard/org__model-a-details'),
            SimpleNamespace(id='open-llm-leaderboard/org__model-b-details'),
        ]
        db = MagicMock()

        result = list(self.make(db)(models))

        self.assertEqual(result, [models[1]])
        self.assertEqual(db.put.call_count, 2)

    def test_load_dataset_is_called_once_for_the_whole_chain(self):
        db = MagicMock()

        with patch.object(list_, 'load_dataset', return_value=self._rows) as mock_load:
            datasets = list_.DatasetIterator('open-llm-leaderboard')
            base = list_.ModelHandler('open-llm-leaderboard')
            flagged = list_.FlaggedHandler(base, datasets)
            list_.DatabaseHandler(flagged, datasets, db)

        mock_load.assert_called_once()

if __name__ == '__main__':
    unittest.main()
