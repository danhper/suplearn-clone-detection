from os import path
import json
import hashlib

import yaml


TRANSFORMER_MAPPING = {
    "FlatVectorIndexASTTransformer": "DFSTransformer",
}


class LanguageConfig:
    def __init__(self, config):
        self.name = config["name"]
        self.vocabulary = path.expandvars(config["vocabulary"])
        self._vocabulary_size = None
        self.embeddings = path.expandvars(config.get("embeddings", ""))
        self.vocabulary_offset = config.get("vocabulary_offset", 0)
        self.input_length = config.get("input_length")
        self.max_length = config.get("max_length")
        self.embeddings_dimension = config["embeddings_dimension"]
        if "output_dimension" in config:
            self.output_dimensions = [config["output_dimension"]]
        else:
            self.output_dimensions = config["output_dimensions"]
        self.transformer_class_name = config.get("transformer_class_name",
                                                 "DFSTransformer")
        if self.transformer_class_name in TRANSFORMER_MAPPING:
            self.transformer_class_name = TRANSFORMER_MAPPING[self.transformer_class_name]
        self.bidirectional_encoding = config.get("bidirectional_encoding", False)
        self.hash_dims = config.get("hash_dims", [])

    @property
    def vocabulary_size(self):
        if self._vocabulary_size is None:
            raise ValueError("vocabulary size needs to be set explicitly")
        return self._vocabulary_size

    @vocabulary_size.setter
    def vocabulary_size(self, value):
        self._vocabulary_size = value


class ModelConfig:
    KNOWN_MERGE_MODES = [
        "simple",
        "bidistance",
        "euclidean_distance",
        "euclidean_similarity",
        "cosine_similarity"
    ]

    def __init__(self, config):
        self.languages = [LanguageConfig(lang) for lang in config["languages"]]
        self.dense_layers = config.get("dense_layers", [64, 64])
        self.optimizer = config.get("optimizer", {"type": "sgd"})
        self.merge_mode = config.get("merge_mode", "simple")
        self.merge_output_dim = config.get("merge_output_dim", 64)
        self.use_output_nn = config.get("use_output_nn", True)
        if not self.merge_mode in self.KNOWN_MERGE_MODES:
            raise ValueError("unknown merge mode: {0}".format(self.merge_mode))
        default_loss = "binary_crossentropy" if self.use_output_nn else "mse"
        self.loss = config.get("loss", default_loss)
        self.metrics = config.get("metrics", ["accuracy"])
        self.normalization_value = 100


class GeneratorConfig:
    def __init__(self, config):
        self.submissions_path = path.expandvars(config["submissions_path"])
        self.asts_path = path.expandvars(config["asts_path"])
        self.filenames_path = None
        self.db_path = path.expandvars(config.get("db_path", ""))
        if "filenames_path " in config:
            self.filenames_path = path.expandvars(config["filenames_path"])
        self.file_format = config.get("file_format", "multi_file")
        self.use_all_combinations = config.get("use_all_combinations", False)
        self.shuffle = config.get("shuffle", True)
        self.shuffle_before_epoch = config.get("shuffle_before_epoch", True)
        self.split_ratio = config.get("split_ratio", [0.8, 0.1, 0.1])
        self.negative_samples = config.get("negative_samples", 1)
        self.negative_sample_candidates = config.get("negative_sample_candidates", 8)
        self.samples_per_problem = config.get("samples_per_problem", 1)
        self.class_weights = config.get("class_weights")
        self.negative_sample_distance = config.get("negative_sample_distance", 0.2)


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

    @property
    def data_generation_config(self):
        return dict(
            languages=[v.name for v in self.model.languages],
            negative_sample_distance=self.generator.negative_sample_distance,
            samples_per_problem=self.generator.samples_per_problem,
            split_ratio=self.generator.split_ratio,
        )

    def data_generation_checksum(self):
        h = hashlib.md5()
        h.update(json.dumps(self.data_generation_config, sort_keys=True).encode("utf-8"))
        return h.hexdigest()

    @classmethod
    def from_file(cls, filepath) -> "Config":
        with open(filepath) as f:
            return cls(yaml.load(f))
