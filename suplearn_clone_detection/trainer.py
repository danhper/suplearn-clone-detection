import yaml

from suplearn_clone_detection.config import Config
from suplearn_clone_detection import ast_transformer
from suplearn_clone_detection.vocabulary import Vocabulary
from suplearn_clone_detection.data_generator import DataGenerator
from suplearn_clone_detection.model import create_model


class Trainer:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = Config(yaml.load(f))
        self.transformers = self._create_transformers()
        self.generator = DataGenerator(self.config.generator, self.transformers)
        self.model = create_model(self.config.model)

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


if __name__ == '__main__':
    trainer = Trainer("config.yml")
    trainer.model.summary()
