from __future__ import annotations

from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker


Base = declarative_base()


def create_engine_and_session(database_url: str):
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    engine = create_engine(database_url, echo=False, future=True, connect_args=connect_args)
    session_local = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    return engine, session_local


@contextmanager
def session_scope(session_local):
    session = session_local()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
