from typing import List, Dict
import time
import functools
import random
import logging

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session, load_only
# from sqlalchemy.sql.expression import func

from suplearn_clone_detection import entities, util
from suplearn_clone_detection.config import Config


def timeit(method):
    def timed(*args, **kwargs):
        ts = time.time()
        result = method(*args, **kwargs)
        te = time.time()

        name = method.__name__
        timeit.log_times.setdefault(name, 0)
        timeit.log_times[name] += int((te - ts) * 1000)
        return result
    return functools.update_wrapper(timed, method)
timeit.log_times = {}


class DatasetGenerator:
    def __init__(self, config: Config, session_maker):
        self.config = config
        self.session_maker = session_maker
        self.config_checksum = self.config.data_generation_checksum()

    @timeit
    def get_positive(self, sess: Session, submission: entities.Submission,
                     lang: str) -> entities.Submission:
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

    @timeit
    def get_negative(self, sess: Session, sample: entities.Sample,
                     lang: str) -> entities.Submission:
        anchor = sample.anchor
        positive = sample.positive
        tokens_diff = int(self.config.generator.negative_sample_distance * positive.tokens_count)
        q = sess.query(entities.Submission) \
                .filter(entities.Submission.tokens_count.between(
                        positive.tokens_count - tokens_diff,
                        positive.tokens_count + tokens_diff)) \
                .filter_by(language_code=lang) \
                .filter((entities.Submission.problem_id != anchor.problem_id) |
                        (entities.Submission.contest_id != anchor.contest_id) |
                        (entities.Submission.contest_type != anchor.contest_type)) \
                .limit(100)
        count = q.count()
        if count == 0:
            return None
        return q.offset(random.randint(0, count)).first()

    def load_submissions(self, sess: Session, lang: str) -> Dict[str, List[entities.Submission]]:
        training_ratio, dev_ratio, _test_ratio = self.config.generator.split_ratio
        submissions = sess.query(entities.Submission) \
                          .filter(entities.Submission.language_code == lang) \
                          .all()
        random.shuffle(submissions)

        training_count = int(len(submissions) * training_ratio)
        dev_count = int(len(submissions) * dev_ratio)
        return dict(
            training=submissions[:training_count],
            dev=submissions[training_count:training_count+dev_count],
            test=submissions[training_count+dev_count:],
        )

    def create_set_samples(self, sess: Session, set_name: str, negative_lang: str,
                           dataset: List[entities.Submission]):
        samples = []
        for i, submission in enumerate(dataset):
            positive = self.get_positive(sess, submission, submission.language_code)
            if not positive:
                continue
            sample = entities.Sample(
                anchor_id=submission.id,
                anchor=submission,
                positive=positive,
                positive_id=positive.id,
                set_name=set_name,
                config_checksum=self.config_checksum,
            )
            negative = self.get_negative(sess, sample, negative_lang)
            if not negative:
                continue
            sample.negative = negative
            sample.negative_id = negative.id
            samples.append(sample)
            if i % 1000 == 0:
                logging.info("%s-%s pairs progress - %s - %s/%s",
                             submission.language_code, negative_lang, set_name, i, len(dataset))
                logging.info("time taken: %s", timeit.log_times)
                timeit.log_times.clear()
        return samples

    def create_lang_samples(self, sess: Session, positive_lang: str,
                            negative_lang: str) -> entities.Sample:
        datasets = self.load_submissions(sess, positive_lang)
        samples = []
        for set_name in datasets:
            dataset = datasets[set_name]
            samples.extend(self.create_set_samples(sess, set_name, negative_lang, dataset))
        return samples

    def check_existing_samples(self, sess: Session):
        q = sess.query(entities.Sample).filter_by(config_checksum=self.config_checksum)
        if q.first():
            raise ValueError("samples already exists for checksum '{0}'. run "
                             "'DELETE FROM samples WHERE config_checksum='{0}' "
                             "if you want to remove them".format(self.config_checksum))

    def create_samples(self):
        with util.session_scope(self.session_maker) as sess:
            self.check_existing_samples(sess)
            languages = [v.name for v in self.config.model.languages]
            samples = self.create_lang_samples(sess, languages[0], languages[1])
            if languages[0] != languages[1]:
                samples.extend(self.create_lang_samples(sess, languages[1], languages[0]))
            sess.bulk_save_objects(samples)
            return len(samples)


def main():
    logging.basicConfig(level=logging.INFO)
    config = Config.from_file("./config.yml")
    engine = create_engine(config.generator.db_path)
    session_maker = sessionmaker(bind=engine)
    dataset_generator = DatasetGenerator(config, session_maker)
    dataset_generator.create_samples()


if __name__ == '__main__':
    main()
