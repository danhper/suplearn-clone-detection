import sys
from typing import Dict, Union
from os import path
import logging

import yaml

from keras.models import load_model, Model
import numpy as np
from sklearn import metrics

from suplearn_clone_detection.layers import custom_objects
from suplearn_clone_detection.dataset.sequences import SuplearnSequence


class Evaluator:
    def __init__(self, model: Union[Model, str]):
        if isinstance(model, str):
            model = load_model(model, custom_objects=custom_objects)
        self.model = model

    def evaluate(self, data: SuplearnSequence, output: str = None,
                 overwrite: bool = False) -> dict:
        labels = data.get_labels()
        logging.info("running predictions with %s samples", len(labels))
        prediction_probs = self.model.predict_generator(data)
        predictions = np.round(prediction_probs)
        precisions, recalls, _ = metrics.precision_recall_curve(labels, predictions)
        results = {
            "samples_count": len(labels),
            "positive_samples_count": len([labels for t in labels if t == 1]),
            "accuracy": float(metrics.accuracy_score(labels, predictions)),
            "precision": float(metrics.precision_score(labels, predictions)),
            "recall": float(metrics.recall_score(labels, predictions)),
            "avg_precision": float(metrics.average_precision_score(labels, predictions)),
            "f1": float(metrics.f1_score(labels, predictions)),
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
