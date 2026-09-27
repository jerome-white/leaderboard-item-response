import unittest
import importlib.util
from pathlib import Path

from mylib import Document, SubmissionInfo

_path = Path(__file__).resolve().parent.parent / 'src' / 'model' / 'aggregate-data.py'
_spec = importlib.util.spec_from_file_location('aggregate_data', _path)
aggregate_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(aggregate_data)

class IndexedCategoryBenchmarkTestCase(unittest.TestCase):
    _info = SubmissionInfo('mmlu', 'algebra', 'org', 'model')
    _documents = [Document('q1', 'algebra'), Document('q2', 'geometry')]

    def test_multitask_understanding_indexes_documents_by_label(self):
        handler = aggregate_data.MultitaskUnderstanding(self._info, self._documents)
        self.assertEqual(handler.subjects, {'q1': 'algebra', 'q2': 'geometry'})

    def test_graduate_level_google_proof_qa_indexes_documents_by_label(self):
        handler = aggregate_data.GraduateLevelGoogleProofQA(self._info, self._documents)
        self.assertEqual(handler.subjects, {'q1': 'algebra', 'q2': 'geometry'})

if __name__ == '__main__':
    unittest.main()
