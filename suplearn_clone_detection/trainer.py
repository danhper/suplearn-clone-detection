import yaml

from suplearn_clone_detection.config import Config
from suplearn_clone_detection import ast_transformer
from suplearn_clone_detection.vocabulary import Vocabulary
from suplearn_clone_detection.data_generator import DataGenerator, LoopBatchIterator
from suplearn_clone_detection.model import create_model


class Trainer:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.raw_config = yaml.load(f)
            self.config = Config(self.raw_config)
        self.transformers = self._create_transformers()
        self.batch_size = self.config.trainer.batch_size
        self.data_generator = None
        self.model = None

    def initialize(self):
        self.data_generator = DataGenerator(self.config.generator, self.transformers)
        self.model = create_model(self.config.model)

    def train(self):
        training_batch_generator = LoopBatchIterator(
            self.data_generator.make_iterator(data_type="training"), self.batch_size)
        dev_batch_generator = LoopBatchIterator(
            self.data_generator.make_iterator(data_type="dev"), self.batch_size)

        self.model.fit_generator(
            training_batch_generator,
            len(training_batch_generator),
            validation_data=dev_batch_generator,
            validation_steps=len(dev_batch_generator),
            epochs=self.config.trainer.epochs)

        if self.config.trainer.output:
            self.model.save(self.config.trainer.output)

    def _create_transformers(self):
        transformers = {}
        for lang in self.config.model.languages:
            vocab = Vocabulary(lang.vocabulary_path)
            lang.vocabulary_size = len(vocab)
            transformer_class = getattr(ast_transformer, lang.transformer_class_name)
            transformer = transformer_class(vocab,
                                            vocabulary_offset=lang.vocabulary_offset,
                                            input_length=lang.input_length)
            transformers[lang.name] = transformer
        return transformers

    def save_results(self):
        pass


def main():
    trainer = Trainer("config.yml")
    print("initializing trainer...")
    trainer.initialize()
    trainer.model.summary()
    trainer.train()


if __name__ == '__main__':
    main()
