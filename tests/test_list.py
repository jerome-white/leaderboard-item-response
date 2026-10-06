import unittest
import importlib.util
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch, MagicMock

from mylib import Backoff, ModelMetadata

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

class ModelMetadataTestCase(unittest.TestCase):
    def test_extracts_fields_from_every_row_regardless_of_flagged_status(self):
        rows = [
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

        with patch.object(list_, 'load_dataset', return_value=rows):
            result = list(list_.model_metadata('open-llm-leaderboard'))

        self.assertEqual(result, [
            ModelMetadata('org', 'model-a', 'chat', 'bfloat16', 7.0, False),
            ModelMetadata('org', 'model-b', 'merge', 'float16', 13.0, True),
        ])

if __name__ == '__main__':
    unittest.main()
