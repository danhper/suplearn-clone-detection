from typing import List
import math
import json

import numpy as np
# from keras.engine import Model
from keras.utils import Sequence
from keras.preprocessing.sequence import pad_sequences

from sqlalchemy.orm import Session, joinedload


from suplearn_clone_detection import entities, ast_transformer
from suplearn_clone_detection.util import memoize
from suplearn_clone_detection.config import Config


class SuplearnSequence(Sequence):
    def __init__(self, set_name: str, config: Config, session_maker) -> None:
        self.config = config
        self.set_name = set_name
        self.session_maker = session_maker
        self.config_checksum = self.config.data_generation_checksum()
        transformers = ast_transformer.create_all(config.model.languages)
        self.ast_transformers = {tr.language: tr for tr in transformers}
        self._session = None
        self._samples_count = 0

    def __enter__(self):
        self._session = self.session_maker()
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self._session.close()
        self._session = None

    def __getitem__(self, index):
        samples = self.get_samples(index)
        lang1_positive, lang2_positive = self.get_positive_pairs(samples)
        lang1_negative, lang2_negative = self.get_negative_pairs(samples)
        lang1_samples = pad_sequences(lang1_positive + lang1_negative)
        lang2_samples = pad_sequences(lang2_positive + lang2_negative)
        labels = np.hstack([np.ones(len(lang1_positive), dtype=np.int32),
                            np.zeros(len(lang1_negative), dtype=np.int32)])
        shuffled_index = np.random.permutation(len(labels))
        X = [lang1_samples[shuffled_index], lang2_samples[shuffled_index]]
        y = labels[shuffled_index]
        return X, y

    def get_positive_pairs(self, samples: List[entities.Sample]):
        lang1 = [self.get_ast(sample.anchor) for sample in samples]
        lang2 = [self.get_ast(sample.positive) for sample in samples]
        return lang1, lang2

    def get_negative_pairs(self, samples: List[entities.Sample]):
        lang1 = [self.get_ast(sample.anchor) for sample in samples]
        lang2 = [self.get_ast(sample.negative) for sample in samples]
        return lang1, lang2

    @memoize
    def get_ast(self, submission: entities.Submission):
        transformer = self.ast_transformers[submission.language_code]
        return transformer.transform_ast(json.loads(submission.ast))

    @property
    def batch_size(self):
        return self.config.trainer.batch_size

    @property
    def db_query(self):
        conditions = dict(set_name=self.set_name,
                          config_checksum=self.config_checksum)
        return self.session \
                   .query(entities.Sample) \
                   .filter_by(**conditions)

    def __len__(self):
        return math.ceil(self.count_samples() * 2 // self.batch_size)

    @memoize
    def get_samples(self, index):
        samples_per_batch = self.batch_size // 2
        offset = samples_per_batch * index
        options = [joinedload(entities.Sample.anchor),
                   joinedload(entities.Sample.positive),
                   joinedload(entities.Sample.negative)]
        return self.db_query.options(*options).offset(offset).limit(samples_per_batch).all()

    @memoize
    def count_samples(self):
        return self.db_query.count()

    @property
    def session(self) -> Session:
        if not self._session:
            raise ValueError("should be used inside 'with'")
        return self._session
