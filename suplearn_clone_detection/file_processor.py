import pickle
import json
import logging
import subprocess
from os import path

from keras.models import load_model

from suplearn_clone_detection.config import Config
from suplearn_clone_detection.ast_loader import ASTLoader
from suplearn_clone_detection.layers import custom_objects, ModelWrapper
from suplearn_clone_detection import ast_transformer


class FileProcessor:
    def __init__(self, config: Config, model: ModelWrapper, options: dict = None):
        if options is None:
            options = {}
        self.options = options
        self.config = config
        self.loader = self._make_ast_loader(config, options)
        self._files_cache = {}
        self.language_names = [lang.name for lang in self.config.model.languages]
        transformers = ast_transformer.create_all(self.config.model.languages)
        self.transformers = {t.language: t for t in transformers}
        self.model = model
        if self.options.get("files_cache"):
            with open(self.options["files_cache"], "rb") as f:
                self._files_cache = pickle.load(f)

    @staticmethod
    def _make_ast_loader(config: Config, options: dict):
        args = dict(
            asts_path=options.get("asts_path") or config.generator.asts_path,
            filenames_path=options.get("filenames_path") or config.generator.filenames_path
        )
        if options.get("file_format"):
            args["file_format"] = options["file_format"]
        return ASTLoader(**args)

    def get_file_vector(self, filename, language):
        if filename not in self._files_cache:
            transformer = self.transformers[language]
            file_ast = self.get_file_ast(filename)
            self._files_cache[filename] = (file_ast, transformer.transform_ast(file_ast))
        return self._files_cache[filename]

    def get_file_ast(self, filename):
        if self.loader.has_file(filename):
            return self.loader.get_ast(filename)
        _, ext = path.splitext(filename)
        executable = "bigcode-astgen-{0}".format(ext[1:])
        logging.warning("%s AST not found, generating with %s", filename, executable)
        res = subprocess.run([executable, filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode != 0:
            raise ValueError("got exit code {0}: {1}".format(res.returncode, res.stderr))
        return json.loads(res.stdout)

    @classmethod
    def from_config(cls, config: Config, model_path: str, options: dict):
        model = load_model(model_path, custom_objects=custom_objects)
        return cls(config, model, options)
