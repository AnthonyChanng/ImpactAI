"""The local SQLite backend queues concurrent sessions instead of deadlocking writes."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

from backend.app.core.database import create_database_engine


def test_sqlite_reader_and_cache_writer_complete_without_database_lock(tmp_path):
    engine = create_database_engine(f"sqlite:///{tmp_path / 'local.db'}")
    attempted, acquired = Event(), Event()
    try:
        with engine.begin() as db:
            db.exec_driver_sql("CREATE TABLE reports (id INTEGER PRIMARY KEY)")
            db.exec_driver_sql("INSERT INTO reports VALUES (1), (2), (3)")
            assert db.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1

        def write():
            attempted.set()
            with engine.begin() as db:
                acquired.set()
                db.exec_driver_sql("INSERT INTO reports VALUES (4)")

        with ThreadPoolExecutor(max_workers=1) as executor:
            with engine.connect() as reader:
                cursor = reader.exec_driver_sql("SELECT id FROM reports ORDER BY id")
                assert cursor.fetchone() == (1,)
                pending = executor.submit(write)
                assert attempted.wait(timeout=2)
                # An unfinished read cursor would block a separate SQLite writer.
                assert not acquired.wait(timeout=0.1)
                cursor.close()
            pending.result(timeout=2)
        assert acquired.is_set()
        with engine.connect() as db:
            assert db.exec_driver_sql("SELECT count(*) FROM reports").scalar() == 4
    finally:
        engine.dispose()
