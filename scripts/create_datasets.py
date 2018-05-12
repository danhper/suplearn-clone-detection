#%%

from typing import List
import random

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, session, load_only

from suplearn_clone_detection import entities
from suplearn_clone_detection.config import Config

config = Config.from_file("./config.yml")

engine = create_engine(config.generator.db_path, echo=True)
Session = sessionmaker(bind=engine)

db_session: session.Session = Session()

#%%

java_submissions = db_session.query(entities.Submission) \
                             .filter(entities.Submission.language_code == "java") \
                             .options(load_only("id", "language_code")) \
                             .all()

random.shuffle(java_submissions)
print(len(java_submissions))

#%%

training_count = int(len(java_submissions) * 0.8)
dev_count = int(len(java_submissions) * 0.1)

java_training_submissions = java_submissions[:training_count]
java_dev_submissions = java_submissions[training_count:training_count+dev_count]
java_test_submissions = java_submissions[training_count+dev_count:]

#%%


def get_positive(sess: session.Session,
                 submission: entities.Submission,
                 languages: List[str]):
    lang = languages[1 if submission.language_code == languages[0] else 0]
    conditions = dict(
        contest_id=submission.contest_id,
        contest_type=submission.contest_type,
        problem_id=submission.problem_id,
        language_code=lang,
    )
    results = sess.query(entities.Submission) \
                  .filter_by(**conditions) \
                  .filter(entities.Submission.id != submission.id) \
                  .all()
    if results:
        return random.choice(results)
    return None

#%%

languages = [v.name for v in config.model.languages]

training_samples = []

for submission in java_training_submissions:
    positive = get_positive(db_session, submission, languages)
    if not positive:
        continue
    sample = entities.Sample(
        anchor_id=submission.id,
        anchor=submission,
        positive=positive,
        positive_id=positive.id,
        set_name="trainig"
    )
    training_samples.append(sample)
