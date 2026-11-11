from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

from backend.app.core.config import settings


def create_database_engine(url):
    options = {}
    if url.startswith("sqlite"):
        # SQLite is the single-process local convenience backend. Serialize its
        # sessions so concurrent readers cannot block coaching-cache commits and
        # stall subsequent requests. PostgreSQL retains its concurrent pool.
        options = {
            "connect_args": {"check_same_thread": False},
            "poolclass": QueuePool,
            "pool_size": 1,
            "max_overflow": 0,
        }
    engine = create_engine(url, pool_pre_ping=True, hide_parameters=True, **options)
    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def enable_foreign_keys(connection, _):
            connection.execute("PRAGMA foreign_keys=ON")

    return engine


engine = create_database_engine(settings.database_url)
SessionLocal = sessionmaker(bind=engine)


def get_db():
    with SessionLocal() as db:
        yield db
