import numpy as np


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
