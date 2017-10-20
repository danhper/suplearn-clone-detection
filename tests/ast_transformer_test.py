import json
import numpy as np

from suplearn_clone_detection.vocabulary import Vocabulary
from suplearn_clone_detection.ast_transformer import FlatVectorIndexASTTransformer


from tests.base import TestCase


class FlatVectorIndexASTTransformerTest(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vocabulary = Vocabulary(cls.fixture_path("vocab-noid.tsv"))
        cls.transformer = FlatVectorIndexASTTransformer({"java": cls.vocabulary})
        with open(cls.fixture_path("asts.json"), "r") as f:
            cls.asts = [json.loads(v) for v in f.read().split("\n") if v]

    def test_transform(self):
        ast = self.asts[0]
        result = self.transformer.transform_ast(ast, "java")
        self.assertEqual(len(ast), len(result))
        for index in result:
            self.assertIsInstance(index, np.int64)
