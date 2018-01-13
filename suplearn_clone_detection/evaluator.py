import sys
from typing import Dict
from os import path
import logging

import yaml

from keras.models import load_model
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from suplearn_clone_detection.layers import custom_objects
from suplearn_clone_detection import ast_transformer
from suplearn_clone_detection.config import Config
from suplearn_clone_detection.data_generator import DataGenerator


class Evaluator:
    def __init__(self, model: "keras.models.Model", data_generator: DataGenerator):
        self.data_generator = data_generator
        self.model = model
        self._inputs = None
        self._targets = None

    def evaluate(self, data_type: str = "dev", output: str = None,
                 overwrite: bool = False,
                 reuse_inputs: bool = False) -> Dict[str, Dict[str, float]]:
        if not reuse_inputs or not self._inputs:
            data_iterator = self.data_generator.make_iterator(data_type=data_type)
            self._inputs, self._targets, _weights = data_iterator.next_batch(len(data_iterator))
        prediction_probs = self.model.predict(self._inputs)
        predictions = np.round(prediction_probs)
        results = {
            "accuracy": float(accuracy_score(self._targets, predictions)),
            "precision": float(precision_score(self._targets, predictions)),
            "recall": float(recall_score(self._targets, predictions)),
            "f1": float(f1_score(self._targets, predictions)),
        }
        if output:
            if path.exists(output) and not overwrite:
                logging.warning("%s exists, skipping", output)
            else:
                with open(output, "w") as f:
                    self.output_results(results, file=f)
        return results

    @staticmethod
    def output_results(results: Dict[str, Dict[str, float]], file=sys.stdout):
        print(yaml.dump(results, default_flow_style=False), file=file, end="")

    @classmethod
    def from_config(cls, config_path: str, model_path: str) -> 'Evaluator':
        config = Config.from_file(config_path)
        transformers = ast_transformer.create_all(config.model.languages)
        data_generator = DataGenerator(config.generator, transformers)
        model = load_model(model_path, custom_objects=custom_objects)
        return cls(model, data_generator)

    @classmethod
    def from_trainer(cls, trainer: 'Trainer') -> 'Evaluator':
        return cls(trainer.model, trainer.data_generator)
