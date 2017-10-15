import numpy as np


class ASTTransformer:
    def transform_ast(self, ast, lang):
        raise NotImplementedError()


class FlatVectorIndexASTTransformer(ASTTransformer):
    def __init__(self, vocabularies):
        self.vocabularies = vocabularies

    def transform_ast(self, ast, lang):
        vocabulary = self.vocabularies[lang]
        indexes = [vocabulary[node] for node in ast]
        return indexes
