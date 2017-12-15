import json
from os import path

class ASTLoader:
    def __init__(self, asts_path, filenames_path=None):
        if filenames_path is None:
            filenames_path = path.splitext(asts_path)[0] + ".txt"
        self._load_asts(asts_path)
        self._load_names(filenames_path)

    def _load_names(self, names_path):
        with open(names_path, "r") as f:
            self.names = {filename.strip(): index for (index, filename) in enumerate(f)}

    def _load_asts(self, asts_path):
        with open(asts_path, "r") as f:
            self.asts = [json.loads(ast) for ast in f]

    def get_ast(self, filename):
        return self.asts[self.names[filename]]

    def has_file(self, filename):
        return filename in self.names
