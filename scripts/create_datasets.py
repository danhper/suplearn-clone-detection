from typing import List, Dict
import random

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, session

from suplearn_clone_detection import entities, util
from suplearn_clone_detection.config import Config



class DatasetGenerator:
    def __init__(self, config: Config, session_maker):
        self.config = config
        self.session_maker = session_maker
        self.session = None  # type: session.Session

    def get_positive(self, submission: entities.Submission, lang: str) -> entities.Submission:
        conditions = dict(
            contest_id=submission.contest_id,
            contest_type=submission.contest_type,
            problem_id=submission.problem_id,
            language_code=lang,
        )
        results = self.session.query(entities.Submission) \
                              .filter_by(**conditions) \
                              .filter(entities.Submission.id != submission.id) \
                              .all()
        if results:
            return random.choice(results)
        return None

    def get_negative(self, sample: entities.Sample, lang: str) -> entities.Submission:
        anchor = sample.anchor
        positive = sample.positive
        tokens_diff = int(self.config.generator.negative_sample_distance * positive.tokens_count)
        results = self.session \
                    .query(entities.Submission) \
                    .filter((entities.Submission.problem_id != anchor.problem_id) |
                            (entities.Submission.contest_id != anchor.contest_id) |
                            (entities.Submission.contest_type != anchor.contest_type)) \
                    .filter_by(language_code=lang) \
                    .filter(entities.Submission.tokens_count.between(
                        positive.tokens_count - tokens_diff,
                        positive.tokens_count + tokens_diff)) \
                    .all()
        if results:
            return random.choice(results)
        return None

    def load_submissions(self, lang: str) -> Dict[str, List[entities.Submission]]:
        training_ratio, dev_ratio, _test_ratio = self.config.generator.split_ratio
        submissions = self.session \
                        .query(entities.Submission) \
                        .filter(entities.Submission.language_code == lang) \
                        .all()
        random.shuffle(submissions)

        training_count = int(len(submissions) * training_ratio)
        dev_count = int(len(submissions) * dev_ratio)
        return {
            "training": submissions[:training_count],
            "dev": submissions[training_count:training_count+dev_count],
            "test": submissions[training_count+dev_count:],
        }


    def create_set_samples(self, set_name: str, negative_lang: str,
                           dataset: List[entities.Submission]):
        samples = []
        for submission in dataset:
            positive = self.get_positive(submission, submission.language_code)
            if not positive:
                continue
            sample = entities.Sample(
                anchor_id=submission.id,
                anchor=submission,
                positive=positive,
                positive_id=positive.id,
                set_name=set_name
            )
            negative = self.get_negative(sample, negative_lang)
            if not negative:
                continue
            sample.negative = negative
            sample.negative_id = negative.id
            samples.append(sample)
        return samples

    def create_lang_samples(self, positive_lang: str, negative_lang: str):
        datasets = self.load_submissions(positive_lang)

        samples = []
        for set_name in datasets:
            dataset = datasets[set_name]
            samples.extend(self.create_set_samples(set_name, negative_lang, dataset))
        return samples

    def create_samples(self):
        with util.session_scope(self.session_maker) as self.session:
            languages = [v.name for v in self.config.model.languages]
            samples = self.create_lang_samples(languages[0], languages[1])
            if languages[0] != languages[1]:
                samples.extend(self.create_lang_samples(languages[1], languages[0]))
            self.session.bulk_save_objects(samples)


def main():
    config = Config.from_file("./config.yml")
    engine = create_engine(config.generator.db_path)
    session_maker = sessionmaker(bind=engine)
    dataset_generator = DatasetGenerator(config, session_maker)
    dataset_generator.create_samples()

    # create_samples(config)


if __name__ == '__main__':
    main()
