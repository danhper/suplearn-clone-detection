import os
from os import path
from datetime import datetime
import logging

import tensorflow as tf
from keras.callbacks import TensorBoard

import yaml

from suplearn_clone_detection.config import Config
from suplearn_clone_detection import database, callbacks, util
from suplearn_clone_detection.dataset.sequences import TrainingSequence, DevSequence
from suplearn_clone_detection.model import create_model


class Trainer:
    def __init__(self, config_path: str, quiet: bool = False):
        with open(config_path) as f:
            self.raw_config = f.read()
            self.config = Config(yaml.load(self.raw_config))
        database.bind_db(self.config.generator.db_path)
        self.model = None
        self.training_data = None
        self.dev_data = None
        self.quiet = quiet

    def initialize(self):
        self.model = create_model(self.config.model)
        graph = tf.get_default_graph()
        self.training_data = TrainingSequence(self.model, graph, self.config)
        self.dev_data = DevSequence(self.config)
        os.makedirs(self.output_dir)
        with open(path.join(self.output_dir, "config.yml"), "w") as f:
            f.write(self.raw_config)
        for transformer in self.training_data.ast_transformers.values():
            vocab = transformer.vocabulary
            vocab.save(self._vocab_path(transformer.language),
                       offset=transformer.vocabulary_offset)

    def train(self):
        logging.info("starting training, outputing to %s", self.output_dir)

        model_path = path.join(self.output_dir, "model-{epoch:02d}.h5")
        results_path = path.join(self.output_dir, "results-dev-{epoch:02d}.yml")

        results_tracker = callbacks.ModelResultsTracker(self.dev_data, self.model)
        checkpoint_callback = callbacks.ModelCheckpoint(
            results_tracker, model_path, save_best_only=True)
        evaluate_callback = callbacks.ModelEvaluator(
            results_tracker, results_path, quiet=self.quiet, save_best_only=True)
        model_callbacks = [checkpoint_callback, evaluate_callback]

        if self.config.trainer.tensorboard_logs:
            tensorboard_logs_path = path.join(self.output_dir, "tf-logs")
            metadata = {}
            for lang_config in self.config.model.languages:
                vocab_path = path.relpath(self._vocab_path(lang_config.name),
                                          tensorboard_logs_path)
                metadata["embedding_{0}".format(lang_config.name)] = vocab_path
            model_callbacks.append(TensorBoard(tensorboard_logs_path))
                                               # TODO: restore embeddings
                                               # embeddings_freq=1,
                                               # embeddings_metadata=metadata))

        self.model.fit_generator(
            self.training_data,
            validation_data=self.dev_data,
            epochs=self.config.trainer.epochs,
            callbacks=model_callbacks)

    def _vocab_path(self, lang):
        return path.join(self.output_dir, "vocab-{0}.tsv".format(lang))

    @property
    @util.memoize
    def output_dir(self):
        output_dir = datetime.now().strftime("%Y%m%d-%H%M")
        return path.join(self.config.trainer.output_dir, output_dir)
