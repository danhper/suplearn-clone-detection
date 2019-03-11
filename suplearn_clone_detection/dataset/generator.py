from typing import List, Dict
import random
import logging

from suplearn_clone_detection import entities
from suplearn_clone_detection.dataset import util
from suplearn_clone_detection.config import Config
from suplearn_clone_detection.database import Session


class DatasetGenerator:
    def __init__(self, config: Config):
        self.config = config
        self.config_checksum = self.config.data_generation_checksum()

    def load_submissions(self, lang: str) -> Dict[str, List[entities.Submission]]:
        training_ratio, dev_ratio, _test_ratio = self.config.generator.split_ratio
        submissions = entities.Submission.query \
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

    def _create_sample(self, dataset_name: str, submission: entities.Submission,
                       sorted_set: List[entities.Submission],
                       positive_samples: List[entities.Submission]):
        if not positive_samples:
            return None, None
        positive_sample_idx = random.randrange(len(positive_samples))
        positive_sample = positive_samples[positive_sample_idx]
        negative_samples = util.select_negative_candidates(
            sorted_set, positive_sample,
            self.config.generator.negative_sample_distance)
        if not negative_samples:
            return None, None
        negative_sample = random.choice(negative_samples)
        return entities.Sample(
            anchor_id=submission.id,
            anchor=submission,
            positive=positive_sample,
            positive_id=positive_sample.id,
            negative=negative_sample,
            negative_id=negative_sample.id,
            dataset_name=dataset_name,
            config_checksum=self.config_checksum,
        ), positive_sample_idx

    def create_set_samples(self, dataset_name: str,
                           lang1_dataset: List[entities.Submission],
                           lang2_dataset: List[entities.Submission]):
        samples = []
        lang2_grouped_dataset = util.group_submissions(lang2_dataset)
        lang2_sorted_dataset = util.sort_dataset(lang2_dataset)
        for i, submission in enumerate(lang1_dataset):
            positive_submissions = lang2_grouped_dataset.get(submission.group_key)
            if not positive_submissions:
                continue
            positive_submissions = positive_submissions.copy()
            for _ in range(self.config.generator.samples_per_problem):
                sample, positive_idx = self._create_sample(dataset_name, submission,
                                                           lang2_sorted_dataset,
                                                           positive_submissions)
                if not sample:
                    break
                del positive_submissions[positive_idx]
                samples.append(sample)
            if i % 1000 == 0 and sample:
                logging.info("%s-%s pairs progress - %s - %s/%s",
                             submission.language_code, sample.negative.language_code,
                             dataset_name, i, len(lang1_dataset))
        return samples

    def create_lang_samples(
            self, lang1_datasets: Dict[str, List[entities.Submission]],
            lang2_datasets: Dict[str, List[entities.Submission]]) -> entities.Sample:
        samples = []
        for dataset_name, dataset in lang1_datasets.items():
            samples.extend(self.create_set_samples(dataset_name, dataset, lang2_datasets[dataset_name]))
        return samples

    def check_existing_samples(self):
        q = entities.Sample.query.filter_by(config_checksum=self.config_checksum)
        if q.first():
            raise ValueError("samples already exists for checksum '{0}'. run "
                             "\"DELETE FROM samples WHERE config_checksum='{0}'\" "
                             "if you want to remove them".format(self.config_checksum))

    def create_samples(self):
        self.check_existing_samples()
        languages = [v.name for v in self.config.model.languages]
        lang1_samples = self.load_submissions(languages[0])
        if languages[0] != languages[1]:
            lang2_samples = self.load_submissions(languages[1])
        else:
            lang2_samples = lang1_samples
        samples = self.create_lang_samples(lang1_samples, lang2_samples)
        if languages[0] != languages[1]:
            samples.extend(self.create_lang_samples(lang2_samples, lang1_samples))
        Session.bulk_save_objects(samples)
        Session.commit()
        return len(samples)
