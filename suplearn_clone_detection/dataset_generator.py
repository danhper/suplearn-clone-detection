from typing import List, Dict, Tuple
import time
import functools
import random
import logging

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session, load_only
# from sqlalchemy.sql.expression import func

from suplearn_clone_detection import entities, util
from suplearn_clone_detection.config import Config


class DatasetGenerator:
    def __init__(self, config: Config, session_maker):
        self.config = config
        self.session_maker = session_maker
        self.config_checksum = self.config.data_generation_checksum()

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

    def group_submissions(self, submissions: List[entities.Submission]) \
            -> Dict[Tuple[str, int, int], entities.Submission]:
        result = {}
        for submission in submissions:
            result.setdefault(submission.group_key, [])
            result[submission.group_key].append(submission)
        return result

    def sort_dataset(self, submissions: List[entities.Submission]) \
            -> Tuple[List[entities.Submission], Dict[int, int]]:
        sorted_submissions = sorted(submissions, key=lambda x: x.tokens_count)
        return sorted_submissions

    def find_submission_index(self, submissions: List[entities.Submission], tokens_count: int, rightmost=False) -> int:
        left = 0
        right = len(submissions)
        while left < right:
            middle = (left + right) // 2
            submission = submissions[middle]
            if submission.tokens_count < tokens_count or \
               (submission.tokens_count == tokens_count and rightmost):
                left = middle + 1
            else:
                right = middle
        return left - 1 if rightmost else left


    def create_set_samples(self, set_name: str,
                           lang1_dataset: List[entities.Submission],
                           lang2_dataset: List[entities.Submission]):
        samples = []
        lang2_grouped_dataset = self.group_submissions(lang2_dataset)
        lang2_sorted_dataset = self.sort_dataset(lang2_dataset)
        for i, submission in enumerate(lang1_dataset):
            positive_samples = lang2_grouped_dataset.get(submission.group_key)
            if not positive_samples:
                continue
            positive_sample = random.choice(positive_samples)
            tok_count = positive_sample.tokens_count
            tokens_diff = int(self.config.generator.negative_sample_distance * tok_count)
            left_index = self.find_submission_index(lang2_sorted_dataset, tok_count - tokens_diff)
            right_index = self.find_submission_index(lang2_sorted_dataset,
                                                     tok_count + tokens_diff, rightmost=True)
            negative_samples = lang2_sorted_dataset[left_index:right_index]
            if not negative_samples:
                continue
            negative_sample = random.choice(negative_samples)
            sample = entities.Sample(
                anchor_id=submission.id,
                anchor=submission,
                positive=positive_sample,
                positive_id=positive_sample.id,
                negative=negative_sample,
                negative_id=negative_sample.id,
                set_name=set_name,
                config_checksum=self.config_checksum,
            )
            samples.append(sample)
            if i % 1000 == 0:
                logging.info("%s-%s pairs progress - %s - %s/%s",
                             submission.language_code, negative_sample.language_code, set_name, i, len(lang1_dataset))
        return samples

    def create_lang_samples(self,
                            lang1_datasets: Dict[str, List[entities.Submission]],
                            lang2_datasets: Dict[str, List[entities.Submission]]) -> entities.Sample:
        # datasets = self.load_submissions(sess, positive_lang)
        samples = []
        for set_name, dataset in lang1_datasets.items():
            samples.extend(self.create_set_samples(set_name, dataset, lang2_datasets[set_name]))
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
            lang1_samples = self.load_submissions(sess, languages[0])
            if languages[0] != languages[1]:
                lang2_samples = self.load_submissions(sess, languages[1])
            else:
                lang2_samples = lang1_samples
            samples = self.create_lang_samples(lang1_samples, lang2_samples)
            if languages[0] != languages[1]:
                samples.extend(self.create_lang_samples(lang2_samples, lang1_samples))
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
