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

@dataclass(frozen=True)
class AuthorModel:
    author: str
    model: str

@dataclass(frozen=True)
class _TaskCategory:
    task: str
    category: str

#
#
#
@dataclass
class EvaluationSet:
    uri: str
    evaluation: str

    def __str__(self):
        path = Path(self.uri, self.evaluation)
        return str(path)

    def get_author_model(self):
        uri = Path(self.uri)

        # parse the values
        (lhs, rhs) = map(uri.name.find, ('_', '__'))
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
            author = uri.name[lhs:]
        else:
            author = uri.name[lhs:_rhs]
        model = None if rhs == _rhs else uri.name[rhs:]

        return AuthorModel(author, model)

    def get_task_category(self):
        (lhs, *body, rhs) = self.evaluation.split('_')
        assert lhs == 'harness', self.evaluation
        assert rhs.isdecimal(), self.evaluation

        name = body.pop(0)
        task = ' '.join(body)
        category = _evaluations.get(name, name)

        return _TaskCategory(task, category)

@dataclass(frozen=True)
class EvaluationInfo(_TaskCategory, AuthorModel):
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

@dataclass(frozen=True)
class LeaderboardResult(EvaluationInfo):
    date: datetime
    prompt: str
    metric: str
    value: float
