from os import path

import yaml


class LanguageConfig:
    def __init__(self, config):
        self.name = config["name"]
        self.vocabulary = path.expandvars(config["vocabulary"])
        self._vocabulary_size = None
        self.embeddings = path.expandvars(config.get("embeddings", ""))
        self.vocabulary_offset = config.get("vocabulary_offset", 0)
        self.input_length = config.get("input_length")
        self.embeddings_dimension = config["embeddings_dimension"]
        self.output_dimension = config["output_dimension"]
        self.transformer_class_name = config.get("transformer_class_name",
                                                 "FlatVectorIndexASTTransformer")
        self.bidirectional_encoding = config.get("bidirectional_encoding", False)

    @property
    def vocabulary_size(self):
        if self._vocabulary_size is None:
            raise ValueError("vocabulary size needs to be set explicitly")
        return self._vocabulary_size

    @vocabulary_size.setter
    def vocabulary_size(self, value):
        self._vocabulary_size = value


class ModelConfig:
    def __init__(self, config):
        self.languages = [LanguageConfig(lang) for lang in config["languages"]]
        self.learning_rate = config.get("learning_rate", 0.01)
        self.dense_layers = config.get("dense_layers", [64, 64])
        self.optimizer = config.get("optimizer", {"type": "sgd"})


class GeneratorConfig:
    def __init__(self, config):
        self.submissions_path = path.expandvars(config["submissions_path"])
        self.asts_path = path.expandvars(config["asts_path"])
        self.filenames_path = None
        if "filenames_path " in config:
            self.filenames_path = path.expandvars(config["filenames_path"])
        self.use_all_combinations = config.get("use_all_combinations", False)
        self.shuffle = config.get("shuffle", True)
        self.shuffle_before_epoch = config.get("shuffle_before_epoch", True)
        self.split_ratio = config.get("split_ratio", [0.8, 0.1, 0.1])
        self.negative_samples = config.get("negative_samples", 1)


class TrainerConfig:
    def __init__(self, config):
        self.epochs = config["epochs"]
        self.batch_size = config.get("batch_size", 128)
        self.output_dir = path.expandvars(config.get("output_dir", ""))
        self.tensorboard_logs = config.get("tensorboard_logs", True)


class Config:
    def __init__(self, config):
        self.model = ModelConfig(config["model"])
        self.generator = GeneratorConfig(config["generator"])
        self.trainer = TrainerConfig(config["trainer"])

    @classmethod
    def from_file(cls, filepath):
        with open(filepath) as f:
            return cls(yaml.load(f))
