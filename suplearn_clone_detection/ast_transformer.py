import sys

import numpy as np

from suplearn_clone_detection.vocabulary import Vocabulary

thismodule = sys.modules[__name__]


def create_all(languages):
    return {lang.name: create(lang) for lang in languages}


def create(language):
    vocab = Vocabulary(language.vocabulary)
    language.vocabulary_size = len(vocab)
    transformer_class = getattr(thismodule, language.transformer_class_name)
    return transformer_class(vocab,
                             vocabulary_offset=language.vocabulary_offset,
                             input_length=language.input_length)


class ASTTransformer:
    def transform_ast(self, ast):
        raise NotImplementedError()


class FlatVectorIndexASTTransformer(ASTTransformer):
    def __init__(self, vocabulary, vocabulary_offset=0, input_length=None):
        self.vocabulary = vocabulary
        self.vocabulary_offset = np.int32(vocabulary_offset)
        self.input_length = input_length

    def transform_ast(self, ast):
        indexes = [self.vocabulary[node] + self.vocabulary_offset for node in ast]
        if not self.input_length:
            return indexes
        if len(indexes) > self.input_length:
            return False
        return indexes + [np.int32(0)] * (self.input_length - len(indexes))
