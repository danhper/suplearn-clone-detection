from typing import Union
from os import path
import random
import itertools
import json

import numpy as np
from sklearn.utils.class_weight import compute_sample_weight

from suplearn_clone_detection.config import GeneratorConfig, Config
from suplearn_clone_detection.ast_loader import ASTLoader
from suplearn_clone_detection import ast_transformer


class DataInput:
    def __init__(self, submission, vector):
        self.submission = submission
        self.vector = vector


class LoopBatchIterator:
    def __init__(self, data_iterator, batch_size):
        self._data_iterator = data_iterator
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def __len__(self):
        return len(self._data_iterator) // self.batch_size

    def __next__(self):
        while True:
            inputs, targets, weights = self._data_iterator.next_batch(self.batch_size)
            if len(targets) < self.batch_size:
                self._data_iterator.reset()
                continue
            return inputs, targets, weights


class DataIterator:
    def __init__(self, config: GeneratorConfig, make_iterator, count):
        self.config = config
        self._make_iterator = make_iterator
        self.reset()
        self._count = count

    def reset(self):
        self._data_iterator = self._make_iterator()

    def iterate(self):
        return self._make_iterator()

    def __len__(self):
        return self._count

    def next_batch(self, batch_size):
        lang1_inputs = []
        lang2_inputs = []
        labels = []
        for _ in range(batch_size):
            try:
                (lang1_input, lang2_input), label = next(self._data_iterator)
            except StopIteration:
                break
            lang1_inputs.append(lang1_input.vector)
            lang2_inputs.append(lang2_input.vector)
            labels.append(label)

        inputs = [np.array(lang1_inputs), np.array(lang2_inputs)]
        targets = np.array(labels).reshape((len(labels), 1))
        weights = compute_sample_weight("balanced", labels)
        if self.config.class_weights:
            class_weights = {k: self.config.class_weights[k] for k in set(labels)}
            weights *= compute_sample_weight(class_weights, labels)

        return inputs, targets, weights


# NOTE: loads all submissions and ASTs in memory
class DataGenerator:
    # pylint: disable=too-many-instance-attributes

    def __init__(self, config, ast_transformers):
        self.ast_loader = ASTLoader(
            config.asts_path, config.filenames_path, config.file_format)
        self.config = config
        self.ast_transformers = {tr.language: tr for tr in  ast_transformers}
        self.languages = [tr.language for tr in ast_transformers]
        self._load_submissions(config.submissions_path)
        self._group_submissions()
        self._split_data()

    def _filename_to_submission(self, filename):
        _basename, ext = path.splitext(filename)
        for known_lang in self.languages:
            if known_lang.startswith(ext[1:]):
                return {"file": filename, "language": known_lang}
        raise ValueError("no language found for {0}".format(filename))

    def load_csv_data(self, csv_data: list):
        lang1_inputs, lang2_inputs, labels = [], [], []
        for datum in csv_data:
            x1, x2, y = datum
            x1, x2 = [self.get_input(self._filename_to_submission(v)) for v in [x1, x2]]
            if x1 and x2:
                lang1_inputs.append(x1)
                lang2_inputs.append(x2)
                labels.append(int(y))

        inputs = [np.array(lang1_inputs), np.array(lang2_inputs)]
        targets = np.array(labels).reshape((len(labels), 1))

        return inputs, targets

    def make_iterator(self, data_type="training"):
        data = getattr(self, "{0}_data".format(data_type))
        data_count = self._count_data(data)
        def make_iterator_func():
            if data_type == "training" and self.config.shuffle_before_epoch:
                random.shuffle(data)
            return self._make_iterator(data)
        return DataIterator(self.config, make_iterator_func, data_count)

    def _load_submissions(self, submissions_filepath):
        self.submissions = []
        with open(submissions_filepath, "r") as f:
            submissions = json.load(f)
        for submission in submissions:
            if self.ast_loader.has_file(submission["file"]):
                self.submissions.append(submission)

    def _group_submissions(self):
        self.submissions_by_language = {lang: [] for lang in self.languages}
        self.submissions_by_problem = {}
        for submission in self.submissions:
            key = (submission["contest_id"], submission["problem_id"])
            current_value = self.submissions_by_problem.get(key, [])
            current_value.append(submission)
            self.submissions_by_problem[key] = current_value
            language = self.normalize_language(submission["language"])
            if language:
                self.submissions_by_language[language].append(submission)
        for submissions in self.submissions_by_language.values():
            submissions.sort(key=lambda sub: len(self.get_ast(sub)))


    def _split_data(self):
        to_split = list(self.submissions_by_problem.values())
        training_ratio, dev_ratio, _test_ratio = self.config.split_ratio
        training_end = int(len(to_split) * training_ratio)
        dev_end = training_end + int(len(to_split) * dev_ratio)
        self.training_data = to_split[:training_end]
        self.dev_data = to_split[training_end:dev_end]
        self.test_data = to_split[dev_end:]

    def _find_ast_index(self, submissions, data_input):
        left, right = 0, len(submissions) - 1
        target_length = len(self.get_ast(data_input.submission))
        while left <= right:
            middle = (left + right) // 2
            current_ast_length = len(self.get_ast(submissions[middle]))
            if current_ast_length == target_length:
                return middle
            elif current_ast_length > target_length:
                right = middle - 1
            else:
                left = middle + 1
        raise ValueError("no ast with length {0} found".format(target_length))

    def _choose_negative_sample(self, submissions, valid_input):
        sample_distance_ratio = self.config.negative_sample_distance
        if sample_distance_ratio < 0:
            return random.choice(submissions)
        sample_distance = int(len(submissions) * sample_distance_ratio)
        input_index = self._find_ast_index(submissions, valid_input)
        sample_index = input_index + random.randint(-sample_distance, sample_distance)
        if sample_index < 0:
            sample_index = 0
        elif sample_index >= len(submissions):
            sample_index = len(submissions) - 1
        return submissions[sample_index]

    def _get_random_input(self, lang, valid_input):
        submissions = self.submissions_by_language[lang]
        random_submission = self._choose_negative_sample(submissions, valid_input)
        random_input = self.get_input(random_submission)
        if random_input:
            return DataInput(random_submission, random_input)
        return False

    def _generate_negative_sample(self, lang1_input, lang2_input):
        if random.random() >= 0.5:
            lang1_input = self._get_random_input(self.languages[0], lang1_input)
        else:
            lang2_input = self._get_random_input(self.languages[1], lang2_input)
        if lang1_input and lang2_input:
            return lang1_input, lang2_input
        return False

    def get_ast(self, submission):
        return self.ast_loader.get_ast(submission["file"])

    def get_input(self, submission):
        lang = self.normalize_language(submission["language"])
        if not lang:
            raise ValueError("language {0} not available".format(submission["language"]))
        ast = self.get_ast(submission)
        return self.transform_ast(ast, lang)

    def transform_ast(self, ast, lang):
        return self.ast_transformers[lang].transform_ast(ast)

    def generate_input(self, lang1_input, lang2_input):
        yield ((lang1_input, lang2_input), 1)
        for _ in range(self.config.negative_samples):
            negative_sample = self._generate_negative_sample(lang1_input, lang2_input)
            if negative_sample:
                yield (negative_sample, 0)

    def _submissions_list_pairs(self, data):
        for submissions in data:
            lang1_submissions = self.filter_language(submissions, self.languages[0])
            lang2_submissions = self.filter_language(submissions, self.languages[1])
            lang1_inputs = self._map_filter_submissions(lang1_submissions)
            lang2_inputs = self._map_filter_submissions(lang2_submissions)
            yield (lang1_inputs, lang2_inputs)

    def _map_filter_submissions(self, submissions):
        result = []
        for submission in submissions:
            transformed_input = self.get_input(submission)
            if transformed_input:
                data_input = DataInput(submission, transformed_input)
                result.append(data_input)
        return result

    def _make_iterator(self, data):
        for (lang1_submissions, lang2_submissions) in self._submissions_list_pairs(data):
            pairs = self._submissions_pairs(lang1_submissions, lang2_submissions)
            for (lang1_sub, lang2_sub) in pairs:
                yield from self.generate_input(lang1_sub, lang2_sub)

    def _submissions_pairs(self, lang1_submissions, lang2_submissions):
        if self.config.use_all_combinations:
            yield from itertools.product(lang1_submissions, lang2_submissions)
        else:
            if len(lang1_submissions) < len(lang2_submissions):
                lang1_submissions = itertools.cycle(lang1_submissions)
            elif len(lang1_submissions) > len(lang2_submissions):
                lang2_submissions = itertools.cycle(lang2_submissions)
            elif self.languages[0] == self.languages[1]:
                lang1_submissions = lang1_submissions[1:]
            yield from zip(lang1_submissions, lang2_submissions)

    def filter_language(self, submissions, language):
        return [s for s in submissions if self.normalize_language(s["language"]) == language]

    def normalize_language(self, language):
        for known_lang in self.languages:
            if language.startswith(known_lang):
                return known_lang
        return None

    def _count_combinations(self, lang1_submissions, lang2_submissions):
        len_lang1 = len(lang1_submissions)
        len_lang2 = len(lang2_submissions)
        if self.config.use_all_combinations:
            return len_lang1 * len_lang2
        if len_lang1 == 0 or len_lang2 == 0:
            return 0
        combinations = max(len_lang1, len_lang2)
        if self.languages[0] == self.languages[1]:
            combinations -= 1
        return combinations

    def _count_data(self, data):
        pairs = self._submissions_list_pairs(data)
        positive_count = sum(self._count_combinations(a, b) for (a, b) in pairs)
        # add negative samples count
        return positive_count + positive_count * self.config.negative_samples

    @classmethod
    def from_config(cls, config: Union[Config, str]) -> "DataGenerator":
        if isinstance(config, str):
            config = Config.from_file(config)
        transformers = ast_transformer.create_all(config.model.languages)
        return cls(config.generator, transformers)
