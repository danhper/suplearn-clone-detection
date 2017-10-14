import random
import itertools
from os import path
import json


# XXX: loads everything in memory
class DataGenerator:
    def __init__(self, submissions_filepath, asts_filepath, names_filepath=None, languages=None):
        if languages is None:
            languages = ("python", "java")
        self.languages = languages
        if names_filepath is None:
            names_filepath = path.splitext(asts_filepath)[0] + ".txt"
        self._load_names(names_filepath)
        self._load_asts(asts_filepath)
        self._load_submissions(submissions_filepath)
        self._group_submissions()
        self.reset()

    def reset(self):
        self._data_iterator = self._make_iterator()

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
        self.submissions_by_language = {self.languages[0]: [], self.languages[1]: []}
        self.submissions_by_problem = {}
        for submission in self.submissions:
            key = (submission["contest_id"], submission["problem_id"])
            current_value = self.submissions_by_problem.get(key, [])
            current_value.append(submission)
            self.submissions_by_problem[key] = current_value
            language = self.normalize_language(submission["language"])
            self.submissions_by_language[language].append(submission)

    def _get_negative_sample(self, lang1_submission, lang2_submission):
        if random.random() >= 0.5:
            lang1_submission = random.choice(self.submissions_by_language[self.languages[0]])
        else:
            lang2_submission = random.choice(self.submissions_by_language[self.languages[1]])
        return (self.get_ast(lang1_submission), self.get_ast(lang2_submission), 0)

    def get_ast(self, submission):
        return self.asts[self.names[submission["file"]]]

    def next_batch(self, batch_size):
        inputs = []
        labels = []
        for _ in range(batch_size):
            try:
                lang1_ast, lang2_ast, label = next(self._data_iterator)
            except StopIteration:
                break
            inputs.append((lang1_ast, lang2_ast))
            labels.append(label)
        return inputs, labels

    def _make_iterator(self):
        for submissions in self.submissions_by_problem.values():
            lang1_submissions = self.filter_language(submissions, self.languages[0])
            lang2_submissions = self.filter_language(submissions, self.languages[1])
            for (lang1_sub, lang2_sub) in itertools.product(lang1_submissions, lang2_submissions):
                yield (self.get_ast(lang1_sub), self.get_ast(lang2_sub), 1)
                yield self._get_negative_sample(lang1_sub, lang2_sub)

    def filter_language(self, submissions, language):
        return [s for s in submissions if self.normalize_language(s["language"]) == language]

    def normalize_language(self, language):
        for known_lang in self.languages:
            if language.startswith(known_lang):
                return known_lang
        raise ValueError("unkown language {0}".format(language))
