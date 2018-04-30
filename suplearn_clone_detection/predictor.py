from typing import Tuple, List

import numpy as np
from tqdm import tqdm

from suplearn_clone_detection.config import Config
from suplearn_clone_detection.file_processor import FileProcessor


class Predictor(FileProcessor):
    def __init__(self, config: Config, model: 'keras.models.Model', options: dict = None):
        super(Predictor, self).__init__(config, model, options)
        self._predictions = []

    def predict(self, files: List[Tuple[str, str]]) -> List[float]:
        batch_size = self.options.get("batch_size") or self.config.trainer.batch_size
        predictions = {}
        for i in tqdm(range(len(files) // batch_size + 1)):
            batch_files = files[i * batch_size:(i + 1) * batch_size]
            to_predict, input_data, assumed_false = self._generate_vectors(batch_files)
            if not to_predict:
                continue
            for file_pair in assumed_false:
                predictions[file_pair] = 0.
            preds = self.model.predict(input_data, batch_size=batch_size)
            for file_pair, pred in zip(to_predict, preds):
                predictions[file_pair] = float(pred[0])
        return self._save_predictions(files, predictions)

    def _below_size_threshold(self, lang1_ast, lang2_ast):
        max_size_diff = self.options.get("max_size_diff")
        if not max_size_diff:
            return True
        size_ratio = len(lang1_ast) / len(lang2_ast)
        return abs(1 - size_ratio) <= self.options["max_size_diff"]

    def _generate_vectors(self, files: List[Tuple[str, str]]) \
            -> Tuple[List[Tuple[str, str]], List[np.array], List[Tuple[str, str]]]:
        lang1_vectors = []
        lang2_vectors = []
        to_predict = []
        assumed_false = []
        for (lang1_file, lang2_file) in files:
            lang1_ast, lang1_vec = self.get_file_vector(lang1_file, self.language_names[0])
            lang2_ast, lang2_vec = self.get_file_vector(lang2_file, self.language_names[1])
            if not lang1_vec or not lang2_vec:
                continue
            if self._below_size_threshold(lang1_ast, lang2_ast):
                lang1_vectors.append(lang1_vec)
                lang2_vectors.append(lang2_vec)
                to_predict.append((lang1_file, lang2_file))
            else:
                assumed_false.append((lang1_file, lang2_file))

        return to_predict, [np.array(lang1_vectors), np.array(lang2_vectors)], assumed_false

    def _save_predictions(self, files, predictions):
        ordered_predictions = []
        for pair in files:
            if pair in predictions:
                ordered_predictions.append((pair, predictions[pair]))
        self._predictions += ordered_predictions
        return ordered_predictions

    @property
    def formatted_predictions(self):
        formatted_predictions = []
        for ((lang1_file, lang2_file), prediction) in self._predictions:
            formatted = "{0},{1},{2}".format(lang1_file, lang2_file, prediction)
            formatted_predictions.append(formatted)
        return "\n".join(formatted_predictions)
