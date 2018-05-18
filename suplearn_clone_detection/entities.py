from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship


Base = declarative_base()


class Submission(Base):
    __tablename__ = "submissions"

    id: int = Column(Integer, primary_key=True)
    contest_id: int = Column(Integer)
    contest_type: str = Column(String)
    problem_id: str = Column(String)
    problem_title: str = Column(String)
    filename: str = Column(String)
    language: str = Column(String)
    language_code: str = Column(String)
    source_length: int = Column(Integer)
    exec_time: int = Column(Integer)
    tokens_count: int = Column(Integer)
    url: str = Column(String)
    ast: str = Column(String)

    @property
    def path(self):
        return "{0}/{1}/{2}/{3}".format(
            self.contest_type,
            self.contest_id,
            self.problem_id,
            self.filename)

    def __repr__(self):
        return "Submission(path=\"{0}\")".format(self.path)


class Sample(Base):
    __tablename__ = "samples"

    id: int = Column(Integer, primary_key=True)
    set_name: str = Column(String)
    config_checksum: str = Column(String)

    anchor_id: int = Column(Integer, ForeignKey("submissions.id"))
    positive_id: int = Column(Integer, ForeignKey("submissions.id"))
    negative_id: int = Column(Integer, ForeignKey("submissions.id"))

    anchor: Submission = relationship("Submission", foreign_keys=[anchor_id])
    positive: Submission = relationship("Submission", foreign_keys=[positive_id])
    negative: Submission = relationship("Submission", foreign_keys=[negative_id])

    def __repr__(self):
        return "Sample(anchor={0}, positive={1}, negative={2})" \
               "set_name=\"{3}\"".format(self.anchor, self.positive,
                                         self.negative, self.set_name)
