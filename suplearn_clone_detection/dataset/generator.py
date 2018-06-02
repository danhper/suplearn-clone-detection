from typing import List, Dict
import random
import logging

from sqlalchemy.orm import Session

from suplearn_clone_detection import entities
from suplearn_clone_detection.dataset import util
from suplearn_clone_detection.util import session_scope
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

    def create_set_samples(self, set_name: str,
                           lang1_dataset: List[entities.Submission],
                           lang2_dataset: List[entities.Submission]):
        samples = []
        lang2_grouped_dataset = util.group_submissions(lang2_dataset)
        lang2_sorted_dataset = util.sort_dataset(lang2_dataset)
        for i, submission in enumerate(lang1_dataset):
            positive_samples = lang2_grouped_dataset.get(submission.group_key)
            if not positive_samples:
                continue
            positive_sample = random.choice(positive_samples)
            negative_samples = util.select_negative_candidates(
                lang2_sorted_dataset, positive_sample,
                self.config.generator.negative_sample_distance)
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
                             submission.language_code, negative_sample.language_code,
                             set_name, i, len(lang1_dataset))
        return samples

    def create_lang_samples(
            self, lang1_datasets: Dict[str, List[entities.Submission]],
            lang2_datasets: Dict[str, List[entities.Submission]]) -> entities.Sample:
        samples = []
        for set_name, dataset in lang1_datasets.items():
            samples.extend(self.create_set_samples(set_name, dataset, lang2_datasets[set_name]))
        return samples

    def check_existing_samples(self, sess: Session):
        q = sess.query(entities.Sample).filter_by(config_checksum=self.config_checksum)
        if q.first():
            raise ValueError("samples already exists for checksum '{0}'. run "
                             "\"DELETE FROM samples WHERE config_checksum='{0}'\" "
                             "if you want to remove them".format(self.config_checksum))

    def create_samples(self):
        with session_scope(self.session_maker) as sess:
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