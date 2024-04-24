from pathlib import Path
from dataclasses import dataclass, asdict

#
#
#
@dataclass
class EvaluationSet:
    uri: Path
    evaluation: str

    def __post_init__(self):
        self.uri = Path(self.uri)

    def __str__(self):
        return str(self.uri)

@dataclass
class EvaluationInfo(EvaluationSet):
    author: str
    model: str

    @classmethod
    def from_evaluation_set(cls, ev_set: EvaluationSet):
        # parse the values
        (lhs, rhs) = map(ev_set.uri.name.find, ('_', '__'))
        (_lhs, _rhs) = (lhs, rhs)

        if lhs < 0:
            raise ValueError(f'Cannot parse name {ev_set.uri}')

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
            author = ev_set.uri.name[lhs:]
        else:
            author = ev_set.uri.name[lhs:_rhs]
        model = None if rhs == _rhs else ev_set.uri.name[rhs:]

        # create the instance
        kwargs = asdict(ev_set)
        return cls(author=author, model=model, **kwargs)
