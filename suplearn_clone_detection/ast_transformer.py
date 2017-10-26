import numpy as np


class ASTTransformer:
    def transform_ast(self, ast, lang):
        raise NotImplementedError()


class FlatVectorIndexASTTransformer(ASTTransformer):
    def __init__(self, vocabularies, base_index=0, input_length=None):
        self.vocabularies = vocabularies
        self.base_index = np.int32(base_index)
        self.input_length = input_length

    def transform_ast(self, ast, lang):
        vocabulary = self.vocabularies[lang]
        indexes = [vocabulary[node] + self.base_index for node in ast]
        if not self.input_length:
            return indexes
        if len(indexes) > self.input_length:
            return False
        return indexes + [np.int32(0)] * (self.input_length - len(indexes))
