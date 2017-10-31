import json
import numpy as np

from suplearn_clone_detection.vocabulary import Vocabulary
from suplearn_clone_detection.ast_transformer import FlatVectorIndexASTTransformer
from tests.base import TestCase


class FlatVectorIndexASTTransformerTest(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vocabulary = Vocabulary(cls.fixture_path("vocab-noid.tsv"))
        with open(cls.fixture_path("asts.json"), "r") as f:
            cls.asts = [json.loads(v) for v in f.read().split("\n") if v]

    def test_simple_transform(self):
        transformer = FlatVectorIndexASTTransformer(self.vocabulary)
        result = transformer.transform_ast(self.asts[0])
        self.assertEqual(len(self.asts[0]), len(result))
        for index in result:
            self.assertIsInstance(index, np.int32)

    def test_transform_with_padding(self):
        transformer = FlatVectorIndexASTTransformer(self.vocabulary, input_length=210)
        padded_result = transformer.transform_ast(self.asts[0]) # length: 206
        self.assertEqual(len(padded_result), 210)
        too_long_result = transformer.transform_ast(self.asts[2]) # length: 215
        self.assertFalse(too_long_result)

    def test_transform_with_base_index(self):
        transformer = FlatVectorIndexASTTransformer(self.vocabulary)
        transformer_with_base_index = FlatVectorIndexASTTransformer(self.vocabulary,
                                                                    vocabulary_offset=2)
        normal_result = transformer.transform_ast(self.asts[0])
        changed_result = transformer_with_base_index.transform_ast(self.asts[0])
        self.assertEqual([v + np.int32(2) for v in normal_result], changed_result)
