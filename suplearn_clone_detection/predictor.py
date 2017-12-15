import logging
import pickle
import subprocess
import json
from os import path
from typing import Tuple, List

import numpy as np
from keras.models import load_model
from tqdm import tqdm

from suplearn_clone_detection import ast_transformer
from suplearn_clone_detection.ast_loader import ASTLoader
from suplearn_clone_detection.layers import custom_objects
from suplearn_clone_detection.config import Config


class Predictor:
    def __init__(self, config: Config, model: 'keras.models.Model', options: dict = None):
        if options is None:
            options = {}
        self.config = config
        self.loader = ASTLoader(config.generator.asts_path, config.generator.filenames_path)
        self.language_names = [lang.name for lang in self.config.model.languages]
        transformers = ast_transformer.create_all(self.config.model.languages)
        self.transformers = {t.language: t for t in transformers}
        self.model = model
        self._files_cache = {}
        self._predictions = []
        self.options = options
        if self.options.get("files_cache"):
            with open(self.options["files_cache"], "rb") as f:
                self._files_cache = pickle.load(f)

    def predict(self, files: List[Tuple[str, str]]) -> List[float]:
        batch_size = self.options.get("batch_size", self.config.trainer.batch_size)
        predictions = []
        for i in tqdm(range(len(files) // batch_size + 1)):
            batch_files = files[i * batch_size:(i + 1) * batch_size]
            to_predict = self._generate_vectors(batch_files)
            predictions += [float(v[0]) for v in
                            self.model.predict(to_predict, batch_size=batch_size)]
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
            formatted = "{0},{1},{2}".format(lang1_file, lang2_file, prediction)
            formatted_predictions.append(formatted)
        return "\n".join(formatted_predictions)

    def get_file_vector(self, filename, language):
        if filename not in self._files_cache:
            transformer = self.transformers[language]
            file_ast = self.get_file_ast(filename)
            self._files_cache[filename] = transformer.transform_ast(file_ast)
        return self._files_cache[filename]

    def get_file_ast(self, filename):
        if self.loader.has_file(filename):
            return self.loader.get_ast(filename)
        _, ext = path.splitext(filename)
        executable = "bigcode-astgen-{0}".format(ext[1:])
        logging.warning("%s AST not found, generating with %s", filename, executable)
        res = subprocess.run([executable, filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode != 0:
            raise ValueError("got exit code {0}: {1}".format(res.returncode, res.stderr))
        return json.loads(res.stdout)

    @classmethod
    def from_config(cls, config_path: str, model_path: str, options: dict) -> 'Predictor':
        config = Config.from_file(config_path)
        model = load_model(model_path, custom_objects=custom_objects)
        return cls(config, model, options)
