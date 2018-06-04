from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.orm.session import Session as DBSession
from sqlalchemy.ext.declarative import declarative_base


Session: DBSession = scoped_session(sessionmaker(autocommit=False, autoflush=False))
Base = declarative_base()
Base.query = Session.query_property()


def bind_db(db_url: str):
    engine = create_engine(db_url)
    Session.configure(bind=engine)
