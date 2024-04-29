from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict

_evaluations = {
    'arc': 'ARC',
    'gsm8k': 'GSM8K',
    'hellaswag': 'HellaSwag',
    'winogrande': 'Winogrande',
    'hendrycksTest': 'MMLU',
    'truthfulqa-mc': 'TruthfulQA',
}

@dataclass
class _AuthorModel:
    author: str
    model: str

@dataclass
class _TaskCategory:
    task: str
    category: str

#
#
#
@dataclass
class EvaluationSet:
    uri: Path
    evaluation: str

    def __str__(self):
        return str(self.uri)

    def get_author_model(self):
        # parse the values
        (lhs, rhs) = map(self.uri.name.find, ('_', '__'))
        (_lhs, _rhs) = (lhs, rhs)

        if lhs < 0:
            raise ValueError(f'Cannot parse name {self.uri}')

        # calculate the bounds
        if lhs == rhs:
            rhs += 2
        elif rhs < 0:
            lhs += 1
        else:
            lhs += 1
            rhs += 2

        # extract the names
        if lhs == _lhs:
            author = None
        elif rhs < 0:
            author = self.uri.name[lhs:]
        else:
            author = self.uri.name[lhs:_rhs]
        model = None if rhs == _rhs else self.uri.name[rhs:]

        return _AuthorModel(author, model)

    def get_task_category(self):
        (lhs, *body, rhs) = self.evaluation.split('_')
        assert lhs == 'harness', self.evaluation
        assert rhs.isdecimal(), self.evaluation

        name = body.pop(0)
        task = ' '.join(body)
        category = _evaluations.get(name, name)

        return _TaskCategory(task, category)

@dataclass
class EvaluationInfo(_TaskCategory, _AuthorModel):
    @classmethod
    def from_evaluation_set(cls, ev_set: EvaluationSet):
        kwargs = {}
        methods = (
            ev_set.get_author_model,
            ev_set.get_task_category,
        )

        for i in methods:
            kwargs.update(asdict(i()))

        return cls(*kwargs)

@dataclass
class LeaderboardResult(EvaluationSet):
    date: datetime
    prompt: str
    metric: str
    value: float
