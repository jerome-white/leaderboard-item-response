import sqlite3
from pathlib import Path
from types import TracebackType
from collections.abc import Iterable, Iterator

from sqlalchemy import Column, Engine, Text, create_engine as _create_engine, event, select
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
    # busy_timeout must be set first: it's what makes a concurrent,
    # lock-contending journal_mode switch wait and retry instead of
    # raising "database is locked" immediately.
    _pragma = {
        'busy_timeout': 5000,
        'journal_mode': 'WAL',
        'synchronous': 'NORMAL',
    }

    def __init__(self, db: Path) -> None:
        self.db = db
        self.connection: sqlite3.Connection | None = None

    def apply_pragma(self, connection: sqlite3.Connection) -> None:
        for (k, v) in self._pragma.items():
            connection.execute(f'PRAGMA {k}={v}')

    def create_engine(self) -> Engine:
        db = self.db.resolve()
        return _create_engine(f'sqlite:///{db}')

    def initialize(self) -> None:
        self.db.parent.mkdir(parents=True, exist_ok=True)

        # Done once, up front, so workers never race each other over
        # creating the schema or switching the (brand new) database
        # into WAL mode for the first time - both need a lock that a
        # concurrent worker startup can otherwise collide on.
        self.connection = sqlite3.connect(self.db)
        try:
            self.apply_pragma(self.connection)
        finally:
            self.connection.close()
            self.connection = None

        engine = self.create_engine()
        try:
            Base.metadata.create_all(engine)
        finally:
            engine.dispose()

class QuestionBankWorker(QuestionBank):
    def __init__(self, db: Path) -> None:
        super().__init__(db)
        self.engine: Engine | None = None
        self.session: SqlAlchemySession | None = None
        self.documents: list[dict[str, str | None]] = []

    def __enter__(self) -> 'QuestionBankWorker':
        if not self.db.parent.is_dir():
            raise FileNotFoundError('Database not initialized')
        self.engine = self.create_engine()

        @event.listens_for(self.engine, 'connect')
        def set_sqlite_pragma(dbapi_connection, connection_record):
            self.apply_pragma(dbapi_connection)

        Base.metadata.create_all(self.engine)
        self.session = SqlAlchemySession(self.engine)

        return self

    def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            traceback: TracebackType | None,
    ) -> None:
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
