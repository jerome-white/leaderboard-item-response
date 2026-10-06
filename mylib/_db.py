import sqlite3
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
from sqlalchemy.dialects.sqlite import insert

from ._dtypes import SubmissionInfo

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

        # expire_on_commit=False: BenchmarkQuestion/ModelMetadata are
        # meant to work as plain value objects everywhere, including
        # after this session closes - without this, commit() (e.g.
        # the implicit one on __exit__) expires every loaded
        # attribute, and touching them post-close raises
        # DetachedInstanceError instead of just returning the value
        # that was already fetched.
        self.session = SqlAlchemySession(self.engine, expire_on_commit=False)

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

    def put(self, values: Iterable) -> None:
        self.values.clear()
        self.values.extend(values)

        if self.values:
            items = list(map(asdict, self.values))
            stmt = (
                insert(self.model)
                .values(items)
                .on_conflict_do_nothing()
            )

            self.session.execute(stmt)
            self.session.commit()

class QuestionDatabase(DatabaseClient):
    def __init__(self, db: Path):
        super().__init__(db, BenchmarkQuestion)

    def get(self, info: SubmissionInfo) -> Iterator[BenchmarkQuestion]:
        stmt = select(self.model).where(
            self.model.benchmark == info.benchmark,
            self.model.subject == info.subject,
        )

        yield from self.session.execute(stmt).scalars()

class ModelDatabase(DatabaseClient):
    def __init__(self, db: Path):
        super().__init__(db, ModelMetadata)

    def get(self) -> Iterator[ModelMetadata]:
        stmt = select(self.model)

        yield from self.session.execute(stmt).scalars()
