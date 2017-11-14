import subprocess
import json
from os import path
from typing import Tuple, List

import numpy as np
from keras.models import load_model

from suplearn_clone_detection.layers import SplitInput
from suplearn_clone_detection import ast_transformer
from suplearn_clone_detection.config import Config


class Predictor:
    def __init__(self, config: Config, model: 'keras.models.Model'):
        self.config = config
        self.language_names = [lang.name for lang in self.config.model.languages]
        self.transformers = ast_transformer.create_all(self.config.model.languages)
        self.model = model
        self._files_cache = {}
        self._predictions = []

    def predict(self, files: List[Tuple[str, str]]) -> List[float]:
        to_predict = self._generate_vectors(files)
        predictions = [float(v[0]) for v in self.model.predict(to_predict)]
        self._save_predictions(files, predictions)
        return predictions

    def _generate_vectors(self, files: List[Tuple[str, str]]) -> List[np.array]:
        lang1_vectors = []
        lang2_vectors = []
        for (lang1_file, lang2_file) in files:
            lang1_vectors.append(self.get_file_vector(lang1_file, self.language_names[0]))
            lang2_vectors.append(self.get_file_vector(lang2_file, self.language_names[1]))
        return [np.array(lang1_vectors), np.array(lang2_vectors)]

    def _save_predictions(self, files, predictions):
        self._predictions += list(zip(files, predictions))

    @property
    def formatted_predictions(self):
        formatted_predictions = []
        for ((lang1_file, lang2_file), prediction) in self._predictions:
            formatted = "{0} - {1}: {2}".format(lang1_file, lang2_file, prediction)
            formatted_predictions.append(formatted)
        return "\n".join(formatted_predictions)

    def get_file_vector(self, filename, language):
        if filename in self._files_cache:
            return self._files_cache[filename]
        transformer = self.transformers[language]
        file_ast = self.get_file_ast(filename)
        return transformer.transform_ast(file_ast)

    @staticmethod
    def get_file_ast(filename):
        _, ext = path.splitext(filename)
        executable = "bigcode-astgen-{0}".format(ext[1:])
        res = subprocess.run([executable, filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode != 0:
            raise ValueError("got exit code {0}: {1}".format(res.returncode, res.stderr))
        return json.loads(res.stdout)

    @classmethod
    def from_config(cls, config_path: str, model_path: str) -> 'Predictor':
        config = Config.from_file(config_path)
        model = load_model(model_path, custom_objects={"SplitInput": SplitInput})
        return cls(config, model)
