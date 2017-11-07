from typing import Dict
from os import path

import yaml

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from suplearn_clone_detection.trainer import Trainer
from suplearn_clone_detection import ast_transformer
from suplearn_clone_detection.config import Config
from suplearn_clone_detection.data_generator import DataGenerator


class Evaluator:
    def __init__(self, config: Config, model: 'keras.models.Model', data_generator: DataGenerator):
        self.config = config
        self.transformers = ast_transformer.create_all(self.config.model.languages)
        self.data_generator = data_generator
        self.batch_size = self.config.trainer.batch_size
        self.model = model

    def evaluate(self, data_type: str = "dev", output: str = None,
                 overwrite: bool = False) -> Dict[str, Dict[str, float]]:
        data_iterator = self.data_generator.make_iterator(data_type=data_type)
        inputs, targets = data_iterator.next_batch(len(data_iterator))
        prediction_probs = self.model.predict(inputs)
        predictions = np.round(prediction_probs)
        results = {data_type: {
            "accuracy": float(accuracy_score(targets, predictions)),
            "precision": float(precision_score(targets, predictions)),
            "recall": float(recall_score(targets, predictions)),
            "f1": float(f1_score(targets, predictions)),
        }}
        if output:
            if path.exists(output) and not overwrite:
                print("{0} exists, skipping".format(output))
            else:
                with open(output, "w") as f:
                    yaml.dump(results, f, default_flow_style=False)
        return results

    @staticmethod
    def output_results(results: Dict[str, Dict[str, float]]):
        print(yaml.dump(results, default_flow_style=False))

    @classmethod
    def from_config(cls, config_path: str, model_path: str) -> 'Evaluator':
        from keras.models import load_model
        config = Config.from_file(config_path)
        transformers = ast_transformer.create_all(config.model.languages)
        data_generator = DataGenerator(config.generator, transformers)
        model = load_model(model_path)
        return cls(config, model, data_generator)

    @classmethod
    def from_trainer(cls, trainer: Trainer) -> 'Evaluator':
        return cls(trainer.config, trainer.model, trainer.data_generator)
