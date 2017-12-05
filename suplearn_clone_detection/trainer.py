import os
from os import path
from datetime import datetime
import logging

from keras.callbacks import TensorBoard

import yaml

from suplearn_clone_detection.config import Config
from suplearn_clone_detection import ast_transformer
from suplearn_clone_detection import callbacks
from suplearn_clone_detection.data_generator import DataGenerator, LoopBatchIterator
from suplearn_clone_detection.model import create_model


class Trainer:
    # pylint: disable=too-many-instance-attributes

    def __init__(self, config_path: str, quiet: bool = False):
        with open(config_path) as f:
            self.raw_config = f.read()
            self.config = Config(yaml.load(self.raw_config))
        self.transformers = ast_transformer.create_all(self.config.model.languages)
        self.batch_size = self.config.trainer.batch_size
        self.data_generator = None
        self.model = None
        self._output_dir = None
        self.quiet = quiet

    def initialize(self):
        self.data_generator = DataGenerator(self.config.generator, self.transformers)
        self.model = create_model(self.config.model)
        os.makedirs(self.output_dir)
        with open(path.join(self.output_dir, "config.yml"), "w") as f:
            f.write(self.raw_config)
        for lang in self.config.model.languages:
            vocab = self.transformers[lang.name].vocabulary
            vocab.save(self._vocab_path(lang), offset=lang.vocabulary_offset)

    def train(self):
        logging.info("starting training, outputing to %s", self.output_dir)

        training_batch_generator = LoopBatchIterator(
            self.data_generator.make_iterator(data_type="training"), self.batch_size)
        dev_batch_generator = LoopBatchIterator(
            self.data_generator.make_iterator(data_type="dev"), self.batch_size)

        model_path = path.join(self.output_dir, "model-{epoch:02d}.h5")
        results_path = path.join(self.output_dir, "results-dev-{epoch:02d}.yml")

        results_tracker = callbacks.ModelResultsTracker(self.data_generator, self.model)
        checkpoint_callback = callbacks.ModelCheckpoint(
            results_tracker, model_path, save_best_only=True)
        evaluate_callback = callbacks.ModelEvaluator(
            results_tracker, results_path, quiet=self.quiet, save_best_only=True)
        model_callbacks = [checkpoint_callback, evaluate_callback]

        if self.config.trainer.tensorboard_logs:
            tensorboard_logs_path = path.join(self.output_dir, "tf-logs")
            metadata = {}
            for lang in self.config.model.languages:
                vocab_path = path.relpath(self._vocab_path(lang), tensorboard_logs_path)
                metadata["embedding_{0}".format(lang.name)] = vocab_path
            model_callbacks.append(TensorBoard(tensorboard_logs_path,
                                               embeddings_freq=1,
                                               embeddings_metadata=metadata))

        self.model.fit_generator(
            training_batch_generator,
            len(training_batch_generator),
            validation_data=dev_batch_generator,
            validation_steps=len(dev_batch_generator),
            epochs=self.config.trainer.epochs,
            callbacks=model_callbacks)

    def _vocab_path(self, lang):
        return path.join(self.output_dir, "vocab-{0}.tsv".format(lang.name))

    @property
    def output_dir(self):
        if self._output_dir:
            return self._output_dir
        output_dir = datetime.now().strftime("%Y%m%d-%H%M")
        self._output_dir = path.join(self.config.trainer.output_dir, output_dir)
        return self._output_dir
