import unittest
import importlib.util
from pathlib import Path
from unittest.mock import patch

_path = Path(__file__).resolve().parent.parent / 'src' / 'data' / 'list_.py'
_spec = importlib.util.spec_from_file_location('list_', _path)
list_ = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(list_)

class DatasetFileSystemTestCase(unittest.TestCase):
    def test_constructs_hffilesystem_with_expand_info(self):
        with patch.object(list_, 'HfFileSystem') as mock_cls:
            list_.DatasetFileSystem(backoff=[1])
            mock_cls.assert_called_once_with(expand_info=True)

if __name__ == '__main__':
    unittest.main()
