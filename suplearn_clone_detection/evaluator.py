import sys
import csv
from typing import Dict
from os import path
import logging

import yaml

from keras.models import load_model
import numpy as np
from sklearn import metrics

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

    def evaluate(self, data_path: str = None, data_type: str = "dev",
                 output: str = None, overwrite: bool = False,
                 reuse_inputs: bool = False) -> dict:
        if not reuse_inputs or not self._inputs:
            self._inputs, self._targets = self._load_data(data_path, data_type)
        logging.info("running predictions with %s samples", len(self._targets))
        prediction_probs = self.model.predict(self._inputs)
        predictions = np.round(prediction_probs)
        precisions, recalls, _ = metrics.precision_recall_curve(self._targets, predictions)
        results = {
            "samples_count": len(self._targets),
            "positive_samples_count": len([self._targets for t in self._targets if t == 1]),
            "accuracy": float(metrics.accuracy_score(self._targets, predictions)),
            "precision": float(metrics.precision_score(self._targets, predictions)),
            "recall": float(metrics.recall_score(self._targets, predictions)),
            "avg_precision": float(metrics.average_precision_score(self._targets, predictions)),
            "f1": float(metrics.f1_score(self._targets, predictions)),
            "pr_curve": dict(precision=precisions.tolist(), recall=recalls.tolist())
        }
        if output:
            if path.exists(output) and not overwrite:
                logging.warning("%s exists, skipping", output)
            else:
                with open(output, "w") as f:
                    self.output_results(results, file=f)
        return results

    def _load_data(self, data_path: str, data_type: str):
        if data_path:
            logging.info("loading data from %s", data_path)
            with open(data_path) as f:
                return self.data_generator.load_csv_data(csv.reader(f))
        else:
            logging.info("generating %s data", data_type)
            data_iterator = self.data_generator.make_iterator(data_type=data_type)
            inputs, targets, _weights = data_iterator.next_batch(len(data_iterator))
            return inputs, targets

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
