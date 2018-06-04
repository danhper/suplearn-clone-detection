from typing import List, Tuple, Iterable
import math
import json
import random
import logging

import numpy as np
from keras.engine import Model
from keras.utils import Sequence
from keras.preprocessing.sequence import pad_sequences

from sqlalchemy.orm import joinedload

from suplearn_clone_detection import entities, ast_transformer, util
from suplearn_clone_detection.util import memoize
from suplearn_clone_detection.config import Config


class SuplearnSequence(Sequence):
    def __init__(self, config: Config) -> None:
        self.config = config
        self.config_checksum = self.config.data_generation_checksum()
        transformers = ast_transformer.create_all(config.model.languages)
        self.ast_transformers = {tr.language: tr for tr in transformers}

    @property
    def dataset_name(self):
        raise NotImplementedError()

    def __getitem__(self, index):
        samples = self.get_samples(index)
        lang1_positive, lang2_positive = self.get_positive_pairs(samples)
        lang1_negative, lang2_negative = self.get_negative_pairs(samples)
        lang1_input = pad_sequences(lang1_positive + lang1_negative)
        lang2_input = pad_sequences(lang2_positive + lang2_negative)
        labels = np.hstack([np.ones(len(lang1_positive), dtype=np.int32),
                            np.zeros(len(lang1_negative), dtype=np.int32)])
        shuffled_index = np.random.permutation(len(labels))
        X = [lang1_input[shuffled_index], lang2_input[shuffled_index]]
        y = labels[shuffled_index]
        return X, y

    def get_positive_pairs(self, samples: List[entities.Sample]) -> Tuple[List[int], List[int]]:
        return self._get_pairs(samples, "positive")

    def get_negative_pairs(self, samples: List[entities.Sample]) -> Tuple[List[int], List[int]]:
        return self._get_pairs(samples, "negative")

    def _get_pairs(self, samples: List[entities.Sample],
                   second_elem_key: str) -> Tuple[List[int], List[int]]:
        lang1 = [self.get_ast(sample.anchor) for sample in samples]
        lang2 = [self.get_ast(getattr(sample, second_elem_key)) for sample in samples]
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
        conditions = dict(dataset_name=self.dataset_name,
                          config_checksum=self.config_checksum)
        return entities.Sample.query.filter_by(**conditions)

    def __len__(self):
        return math.ceil(self.count_samples() * 2 / self.batch_size)

    @memoize
    def get_samples(self, index):
        if index >= len(self):
            raise IndexError("sequence index out of range")
        samples_per_batch = self.batch_size // 2
        offset = samples_per_batch * index
        options = [joinedload(entities.Sample.anchor),
                   joinedload(entities.Sample.positive),
                   joinedload(entities.Sample.negative)]
        return self.db_query.options(*options).offset(offset).limit(samples_per_batch).all()

    @memoize
    def count_samples(self):
        return self.db_query.count()


class TrainingSequence(SuplearnSequence):
    def __init__(self, model: Model, config: Config) -> None:
        super(TrainingSequence, self).__init__(config)
        self.model = model

    def get_negative_pairs(self, samples: List[entities.Sample]) \
            -> Tuple[List[List[int]], List[List[int]]]:
        count_per_anchor = self.config.generator.negative_sample_candidates
        anchors = [sample.anchor for sample in samples]
        anchor_asts = [self.get_ast(anchor) for anchor in anchors]
        candidates = random.sample(self.candidates_pool(samples[0].negative.language_code),
                                   count_per_anchor * len(samples))
        candidate_asts = [self.get_ast(submission) for submission in candidates]

        # input lenghts: len(samples) * count_per_anchor
        lang1_input = pad_sequences([ast for ast in anchor_asts for _ in range(count_per_anchor)])
        lang2_input = pad_sequences([ast for ast in candidate_asts])

        predictions = self.model.predict([lang1_input, lang2_input], batch_size=len(lang1_input))
        negative_asts = self._collect_negative_asts(anchors, candidates,
                                                    candidate_asts, predictions)

        return anchor_asts, negative_asts

    def _collect_negative_asts(self, anchors: List[entities.Submission],
                               candidates: List[entities.Submission],
                               candidate_asts: List[List[int]],
                               predictions: Iterable[int]) -> List[List[int]]:
        count_per_anchor = self.config.generator.negative_sample_candidates
        negative_asts = []
        for i, sample_predictions in enumerate(util.in_batch(predictions, count_per_anchor)):
            negative_ast = None
            max_prediction = -1
            base_index = i * count_per_anchor
            for j, prediction in enumerate(sample_predictions):
                if prediction > max_prediction and \
                        candidates[base_index + j].group_key != anchors[i].group_key:
                    negative_ast = candidate_asts[base_index + j]
                    max_prediction = prediction

            if not negative_ast:
                logging.warning("could not find a valid negative sample")
                negative_ast = candidates[base_index]

            negative_asts.append(negative_ast)
        return negative_asts

    @memoize
    def candidates_pool(self, language_code: str):
        return entities.Submission.query.filter_by(language_code=language_code).all()

    @property
    def dataset_name(self):
        return "training"


class DevSequence(SuplearnSequence):
    @property
    def dataset_name(self):
        return "dev"


class TestSequence(SuplearnSequence):
    @property
    def dataset_name(self):
        return "test"
