import sys
from typing import Dict
from os import path
import logging

import yaml

from keras.models import load_model
import numpy as np
from sklearn import metrics
from keras.utils import Sequence

from suplearn_clone_detection.layers import custom_objects
from suplearn_clone_detection.config import Config
from suplearn_clone_detection.dataset.sequences import DevSequence


class Evaluator:
    def __init__(self, model: "keras.models.Model", data: Sequence):
        self.data = data
        self.model = model
        self._targets = None

    def evaluate(self, data_path: str = None, data_type: str = "dev",
                 output: str = None, overwrite: bool = False,
                 reuse_inputs: bool = False) -> dict:
        logging.info("running predictions with %s samples", len(self._targets))
        prediction_probs = self.model.predict_generator(self.data)
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

    @staticmethod
    def output_results(results: Dict[str, Dict[str, float]], file=sys.stdout):
        print(yaml.dump(results, default_flow_style=False), file=file, end="")

    @classmethod
    def from_config(cls, config: Config, model_path: str) -> 'Evaluator':
        data = DevSequence(config)
        model = load_model(model_path, custom_objects=custom_objects)
        return cls(model, data)

    @classmethod
    def from_trainer(cls, trainer: 'Trainer') -> 'Evaluator':
        return cls(trainer.model, trainer.dev_data)
