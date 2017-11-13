from typing import Type, Dict
import sys

import numpy as np

from suplearn_clone_detection import ast
from suplearn_clone_detection.vocabulary import Vocabulary

thismodule = sys.modules[__name__]


class ASTTransformer:
    def __init__(self, vocabulary, vocabulary_offset=0, input_length=None):
        self.vocabulary = vocabulary
        self.vocabulary_offset = np.int32(vocabulary_offset)
        self.input_length = input_length
        self.total_input_length = input_length

    def transform_ast(self, list_ast):
        raise NotImplementedError()

    @property
    def split_input(self):
        return False

    def node_index(self, node):
        return self.vocabulary[node] + self.vocabulary_offset

    def pad(self, indexes, pad_value=np.int32(0)):
        return indexes + [pad_value] * (self.input_length - len(indexes))


class DFSTransformer(ASTTransformer):
    def transform_ast(self, list_ast):
        indexes = [self.node_index(node) for node in list_ast]
        if not self.input_length:
            return indexes
        if len(indexes) > self.input_length:
            return False
        return self.pad(indexes)


class BiDFSTransformer(ASTTransformer):
    def __init__(self, vocabulary, vocabulary_offset=0, input_length=None):
        super(BiDFSTransformer, self).__init__(vocabulary, vocabulary_offset, input_length)
        if self.total_input_length:
            self.total_input_length *= 2

    def transform_ast(self, list_ast):
        ast_root = ast.from_list(list_ast)
        forward_indexes = [self.node_index(node) for node in ast_root.dfs()]
        backward_indexes = [self.node_index(node) for node in ast_root.dfs(reverse=True)]
        if not self.input_length:
            return forward_indexes + backward_indexes
        if len(forward_indexes) > self.input_length:
            return False
        return self.pad(forward_indexes) + self.pad(backward_indexes)

    @property
    def split_input(self):
        return True


def get_class(language) -> Type[ASTTransformer]:
    return getattr(thismodule, language.transformer_class_name)


def create_all(languages) -> Dict[str, ASTTransformer]:
    return {lang.name: create(lang) for lang in languages}


def create(language) -> ASTTransformer:
    vocab = Vocabulary(language.vocabulary)
    language.vocabulary_size = len(vocab)
    transformer_class = get_class(language)
    return transformer_class(vocab,
                             vocabulary_offset=language.vocabulary_offset,
                             input_length=language.input_length)
