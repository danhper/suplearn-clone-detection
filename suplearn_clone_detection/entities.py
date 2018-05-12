from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship


Base = declarative_base()


class Submission(Base):
    __tablename__ = "submissions"

    id = Column(Integer, primary_key=True)
    contest_id = Column(Integer)
    contest_type = Column(String)
    problem_id = Column(String)
    problem_title = Column(String)
    filename = Column(String)
    language = Column(String)
    language_code = Column(String)
    source_length = Column(Integer)
    exec_time = Column(Integer)
    tokens_count = Column(Integer)
    url = Column(String)
    ast = Column(String)

    @property
    def path(self):
        return "{0}/{1}/{2}/{3}".format(
            self.contest_type,
            self.contest_id,
            self.problem_id,
            self.filename)

    def __repr__(self):
        return "Submission(path=\"{0}\")".format(self.path)


class TrainingSample(Base):
    __tablename__ = "training_samples"

    id = Column(Integer, primary_key=True)

    anchor_id = Column(Integer, ForeignKey("submissions.id"))
    positive_id = Column(Integer, ForeignKey("submissions.id"))
    negative_id = Column(Integer, ForeignKey("submissions.id"))

    anchor = relationship("Submission", foreign_keys=[anchor_id])
    positive = relationship("Submission", foreign_keys=[positive_id])
    negative = relationship("Submission", foreign_keys=[negative_id])
