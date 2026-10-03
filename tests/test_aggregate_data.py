import queue
import tempfile
import unittest
import importlib.util
import pandas as pd
from pathlib import Path
from types import SimpleNamespace

from mylib import Document, Experiment, QuestionBank, QuestionBankWorker, SubmissionInfo

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
    def test_matches_numeric_document_ids_against_the_question_bank(self):
        # MMLU-Pro's doc_ids are plain digit strings ('0', '1', ...),
        # stored as TEXT in the question bank. pandas infers an
        # all-numeric CSV column as int64 unless told otherwise, so
        # without dtype enforcement this lookup silently breaks: int
        # 0 and str '0' never compare equal as dict keys.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            qbank_path = tmp.joinpath('questions.sqlite')
            QuestionBank(qbank_path).initialize()

            info = SubmissionInfo('mmlu', 'pro', 'org', 'model-x')
            with QuestionBankWorker(qbank_path) as db:
                db.put(info, [Document('0', 'physics'), Document('1', 'law')])

            data_root = tmp.joinpath('responses')
            path = data_root.joinpath(info.to_path('.csv.gz'))
            path.parent.mkdir(parents=True)
            pd.DataFrame({
                'author': ['org', 'org'],
                'model': ['model-x', 'model-x'],
                'document': [0, 1],
                'metric': ['acc', 'acc'],
                'score': [1.0, 0.0],
            }).to_csv(path, index=False, compression='gzip')

            experiment = Experiment('mmlu', 'physics', ['physics'])
            args = SimpleNamespace(data_root=data_root, question_bank=qbank_path)

            incoming = _BoundedQueue()
            incoming.put(path)
            outgoing = queue.Queue()

            with self.assertRaises(_BoundedQueue.Stop):
                aggregate_data.func(incoming, outgoing, experiment, args)

            records = []
            while not outgoing.empty():
                item = outgoing.get()
                if item:
                    records.extend(item)

        self.assertEqual(
            records,
            [aggregate_data.Record('org', 'model-x', '0', 1.0)],
        )

if __name__ == '__main__':
    unittest.main()
