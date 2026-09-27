from pathlib import Path
from collections.abc import Iterable, Iterator

from sqlalchemy import Column, Text, create_engine, event, select
from sqlalchemy.orm import Session as SqlAlchemySession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.sqlite import insert

from ._dtypes import Document, SubmissionInfo

Base = declarative_base()
class BenchmarkQuestion(Base):
    __tablename__ = 'benchmark_questions'

    benchmark = Column(Text, primary_key=True, nullable=False)
    subject   = Column(Text, primary_key=True, nullable=False)
    doc_id    = Column(Text, primary_key=True, nullable=False)
    label     = Column(Text)

class QuestionBank:
    _pragma = {
        'journal_mode': 'WAL',
        'busy_timeout': 5000,
        'synchronous': 'NORMAL',
    }

    def __init__(self, db: Path):
        self.db = db
        self.engine = None
        self.connection = None
        self.documents = []

    def __enter__(self):
        self.db.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f'sqlite:///{self.db.resolve()}')

        @event.listens_for(self.engine, 'connect')
        def set_sqlite_pragma(dbapi_connection, connection_record):
            for (k, v) in self._pragma.items():
                dbapi_connection.execute(f'PRAGMA {k}={v}')

        Base.metadata.create_all(self.engine)
        self.session = SqlAlchemySession(self.engine)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.session:
            try:
                if exc_type is None:
                    self.session.commit()
                else:
                    self.session.rollback()
            finally:
                self.session.close()

        if self.engine:
            self.engine.dispose()

    def get(self, info: SubmissionInfo) -> Iterator[Document]:
        stmt = (
            select(
                BenchmarkQuestion.doc_id,
                BenchmarkQuestion.label,
            )
            .where(
                BenchmarkQuestion.benchmark == info.benchmark,
                BenchmarkQuestion.subject == info.subject
            )
        )

        for row in self.session.execute(stmt):
            yield Document(row.doc_id, row.label)

    def put(self, info: SubmissionInfo, documents: Iterable[Document]) -> None:
        self.documents.clear()
        for doc in documents:
            self.documents.append({
                'benchmark': info.benchmark,
                'subject': info.subject,
                'doc_id': doc.question,
                'label': doc.label,
            })

        if self.documents:
            stmt = (
                insert(BenchmarkQuestion)
                .values(self.documents)
                .on_conflict_do_nothing()
            )

            self.session.execute(stmt)
            self.session.commit()
