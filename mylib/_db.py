import sqlite3
from types import TracebackType
from pathlib import Path
from dataclasses import asdict
from collections.abc import Iterable, Iterator

from sqlalchemy import (
    Boolean,
    Column,
    Engine,
    Float,
    Integer,
    Text,
    create_engine,
    event,
    select,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    MappedAsDataclass,
    Session as SqlAlchemySession,
    mapped_column,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.sqlite import insert

from ._dtypes import Document, ModelInfo, SubmissionInfo

#
#
#
class Base(MappedAsDataclass, DeclarativeBase):
    pass

class BenchmarkQuestion(Base):
    __tablename__ = 'questions'

    benchmark: Mapped[str] = mapped_column(Text, primary_key=True)
    subject:   Mapped[str] = mapped_column(Text, primary_key=True)
    doc_id:    Mapped[int] = mapped_column(Integer, primary_key=True)
    label:     Mapped[str | None] = mapped_column(Text, default=None)

class ModelMetadata(Base):
    __tablename__ = 'models'

    author:    Mapped[str] = mapped_column(Text, primary_key=True)
    model:     Mapped[str] = mapped_column(Text, primary_key=True)
    mtype:     Mapped[str | None] = mapped_column(Text, default=None)
    precision: Mapped[str | None] = mapped_column(Text, default=None)
    params:    Mapped[float | None] = mapped_column(Float, default=None)
    merged:    Mapped[bool | None] = mapped_column(Boolean, default=None)

#
#
#
class LeaderboardDatabase:
    # busy_timeout must be set first: it's what makes a concurrent,
    # lock-contending journal_mode switch wait and retry instead of
    # raising "database is locked" immediately.
    _pragma = {
        'busy_timeout': 5000,
        'journal_mode': 'WAL',
        'synchronous': 'NORMAL',
    }

    def __init__(self, db: Path):
        self.db = db

    def apply_pragma(self, connection: sqlite3.Connection) -> None:
        for (k, v) in self._pragma.items():
            connection.execute(f'PRAGMA {k}={v}')

    def create_engine(self) -> Engine:
        db = self.db.resolve()
        return create_engine(f'sqlite:///{db}')

    def initialize(self) -> None:
        self.db.parent.mkdir(parents=True, exist_ok=True)

        # Done once, up front, so workers never race each other over
        # creating the schema or switching the (brand new) database
        # into WAL mode for the first time - both need a lock that a
        # concurrent worker startup can otherwise collide on.
        connection = sqlite3.connect(self.db)
        try:
            self.apply_pragma(connection)
        finally:
            connection.close()

        engine = self.create_engine()
        try:
            Base.metadata.create_all(engine)
        finally:
            engine.dispose()

#
#
#
class DatabaseClient(LeaderboardDatabase):
    def __init__(self, db: Path, model: Base):
        super().__init__(db)

        self.model = model
        self.engine = None
        self.session = None
        self.values = []

    def __enter__(self):
        if not self.db.parent.is_dir():
            raise FileNotFoundError('Database not initialized')
        self.engine = self.create_engine()

        @event.listens_for(self.engine, 'connect')
        def set_sqlite_pragma(dbapi_connection, connection_record):
            self.apply_pragma(dbapi_connection)

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

    def get(self, *args, **kwargs) -> Iterator:
        raise NotImplementedError()

    def put(self, values: Iterable, *args, **kwargs) -> None:
        self.values.clear()
        self.values.extend(self.gather(values, *args, **kwargs))

        if self.values:
            items = list(map(asdict, self.values))
            stmt = (
                insert(self.model)
                .values(items)
                .on_conflict_do_nothing()
            )

            self.session.execute(stmt)
            self.session.commit()

    def gather(self, values: Iterable, *args, **kwargs) -> None:
        raise NotImplementedError()

class QuestionDatabase(DatabaseClient):
    def __init__(self, db: Path):
        super().__init__(db, BenchmarkQuestion)

    def get(self, info: SubmissionInfo) -> Iterator[Document]:
        stmt = (
            select(
                self.model.doc_id,
                self.model.label,
            )
            .where(
                self.model.benchmark == info.benchmark,
                self.model.subject == info.subject
            )
        )

        for row in self.session.execute(stmt):
            yield Document(row.doc_id, row.label)

    def gather(
            self,
            values: Iterable[Document],
            info: SubmissionInfo,
    ) -> Iterator[Base]:
        for doc in values:
            yield BenchmarkQuestion(
                benchmark=info.benchmark,
                subject=info.subject,
                doc_id=doc.question,
                label=doc.label,
            )

class ModelDatabase(DatabaseClient):
    def __init__(self, db: Path):
        super().__init__(db, ModelMetadata)

    def get(self) -> Iterator[ModelInfo]:
        stmt = select(self.model)
        for row in self.session.execute(stmt).scalars():
            yield ModelInfo(*row)

    def gather(self, values: Iterable[ModelInfo]) -> Iterator[Base]:
        for model in values:
            yield ModelMetadata(
                author=model.author,
                model=model.model,
                mtype=model.mtype,
                precision=model.precision,
                params=model.params,
                merged=model.merged,
            )
