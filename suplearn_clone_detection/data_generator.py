import random
import itertools
from os import path
import json

import numpy as np


# XXX: loads everything in memory
class DataGenerator:
    def __init__(self, config, ast_transformers):
        if config.filenames_path is None:
            filenames_path = path.splitext(config.asts_path)[0] + ".txt"
        self.ast_transformers = ast_transformers
        self.languages = list(ast_transformers.keys())
        self._load_names(filenames_path)
        self._load_asts(config.asts_path)
        self._load_submissions(config.submissions_path)
        self._group_submissions()
        self._use_all_combinations = config.use_all_combinations
        self._count = self._count_data()
        self.reset()

    def __len__(self):
        return self._count

    def reset(self):
        self._data_iterator = self.make_iterator()

    def _load_names(self, names_filepath):
        with open(names_filepath, "r") as f:
            self.names = {filename.strip(): index for (index, filename) in enumerate(f)}

    def _load_asts(self, asts_filepath):
        with open(asts_filepath, "r") as f:
            self.asts = [json.loads(ast) for ast in f]

    def _load_submissions(self, submissions_filepath):
        self.submissions = []
        with open(submissions_filepath, "r") as f:
            submissions = json.load(f)
        for submission in submissions:
            if submission["file"] in self.names:
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
            self.submissions_by_language[language].append(submission)

    def _generate_negative_sample(self, lang1_input, lang2_input):
        if random.random() >= 0.5:
            submissions = self.submissions_by_language[self.languages[0]]
            lang1_input = self.get_input(random.choice(submissions))
        else:
            submissions = self.submissions_by_language[self.languages[1]]
            lang2_input = self.get_input(random.choice(submissions))
        if lang1_input and lang2_input:
            return lang1_input, lang2_input
        return False

    def get_ast(self, submission):
        return self.asts[self.names[submission["file"]]]

    def get_input(self, submission):
        lang = self.normalize_language(submission["language"])
        ast = self.get_ast(submission)
        return self.ast_transformers[lang].transform_ast(ast)

    def generate_input(self, lang1_input, lang2_input):
        negative_sample = self._generate_negative_sample(lang1_input, lang2_input)
        yield ((lang1_input, lang2_input), 1)
        if negative_sample:
            yield (negative_sample, 0)

    def next_batch(self, batch_size):
        lang1_inputs = []
        lang2_inputs = []
        labels = []
        for _ in range(batch_size):
            try:
                (lang1_input, lang2_input), label = next(self._data_iterator)
            except StopIteration:
                break
            lang1_inputs.append(lang1_input)
            lang2_inputs.append(lang2_input)
            labels.append([label])
        return [np.array(lang1_inputs), np.array(lang2_inputs)], np.array(labels)

    def _submissions_list_pairs(self):
        for submissions in self.submissions_by_problem.values():
            lang1_submissions = self.filter_language(submissions, self.languages[0])
            lang2_submissions = self.filter_language(submissions, self.languages[1])
            lang1_inputs = self._map_filter_submissions(lang1_submissions)
            lang2_inputs = self._map_filter_submissions(lang2_submissions)
            yield (lang1_inputs, lang2_inputs)

    def _map_filter_submissions(self, submisisons):
        result = []
        for submission in submisisons:
            transformed_input = self.get_input(submission)
            if transformed_input:
                result.append(transformed_input)
        return result

    def _count_data(self):
        # NOTE: multiply by 2 to add the negative sample
        return sum(self._count_combinations(a, b) for (a, b) in self._submissions_list_pairs()) * 2

    def _count_combinations(self, lang1_submissions, lang2_submissions):
        len_lang1 = len(lang1_submissions)
        len_lang2 = len(lang2_submissions)
        if self._use_all_combinations:
            return len_lang1 * len_lang2
        if len_lang1 == 0 or len_lang2 == 0:
            return 0
        return max(len_lang1, len_lang2)

    def make_iterator(self):
        for (lang1_submissions, lang2_submissions) in self._submissions_list_pairs():
            pairs = self._submissions_pairs(lang1_submissions, lang2_submissions)
            for (lang1_sub, lang2_sub) in pairs:
                yield from self.generate_input(lang1_sub, lang2_sub)

    def _submissions_pairs(self, lang1_submissions, lang2_submissions):
        if self._use_all_combinations:
            yield from itertools.product(lang1_submissions, lang2_submissions)
        else:
            if len(lang1_submissions) < len(lang2_submissions):
                lang1_submissions = itertools.cycle(lang1_submissions)
            elif len(lang1_submissions) > len(lang2_submissions):
                lang2_submissions = itertools.cycle(lang2_submissions)
            yield from zip(lang1_submissions, lang2_submissions)

    def filter_language(self, submissions, language):
        return [s for s in submissions if self.normalize_language(s["language"]) == language]

    def normalize_language(self, language):
        for known_lang in self.languages:
            if language.startswith(known_lang):
                return known_lang
        raise ValueError("unkown language {0}".format(language))
