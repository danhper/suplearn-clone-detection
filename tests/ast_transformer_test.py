import json
import numpy as np

from suplearn_clone_detection.vocabulary import Vocabulary
from suplearn_clone_detection.ast_transformer import DFSTransformer, BiDFSTransformer
from tests.base import TestCase


class TransformerTestCase(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vocabulary = Vocabulary.from_file(cls.fixture_path("vocab-noid.tsv"))
        with open(cls.fixture_path("asts.json"), "r") as f:
            cls.asts = [json.loads(v) for v in f.read().split("\n") if v]


class DFSTransformerTest(TransformerTestCase):
    def test_split_input(self):
        self.assertFalse(DFSTransformer("lang", self.vocabulary).split_input)

    def test_input_length(self):
        transformer = DFSTransformer("lang", self.vocabulary, input_length=210)
        self.assertEqual(transformer.total_input_length, 210)

    def test_simple_transform(self):
        transformer = DFSTransformer("lang", self.vocabulary)
        result = transformer.transform_ast(self.asts[0])
        self.assertEqual(len(self.asts[0]), len(result))
        for index in result:
            self.assertIsInstance(index, np.int32)

    def test_transform_with_padding(self):
        transformer = DFSTransformer("lang", self.vocabulary, input_length=210)
        padded_result = transformer.transform_ast(self.asts[0]) # length: 206
        self.assertEqual(len(padded_result), 210)
        too_long_result = transformer.transform_ast(self.asts[2]) # length: 215
        self.assertFalse(too_long_result)

    def test_transform_with_base_index(self):
        transformer = DFSTransformer("lang", self.vocabulary)
        transformer_with_base_index = DFSTransformer("lang", self.vocabulary, vocabulary_offset=2)
        normal_result = transformer.transform_ast(self.asts[0])
        changed_result = transformer_with_base_index.transform_ast(self.asts[0])
        self.assertEqual([v + np.int32(2) for v in normal_result], changed_result)


class BiDFSTransformerTest(TransformerTestCase):
    def test_split_input(self):
        self.assertTrue(BiDFSTransformer("lang", self.vocabulary).split_input)

    def test_input_length(self):
        transformer = BiDFSTransformer("lang", self.vocabulary, input_length=210)
        self.assertEqual(transformer.total_input_length, 420)

    def test_simple_transform(self):
        transformer = BiDFSTransformer("lang", self.vocabulary)
        result = transformer.transform_ast(self.asts[0])
        self.assertEqual(len(self.asts[0]) * 2, len(result))
        for index in result:
            self.assertIsInstance(index, np.int32)

    def test_transform_with_padding(self):
        transformer = BiDFSTransformer(
            "lang", self.vocabulary, input_length=210, vocabulary_offset=1)
        padded_result = transformer.transform_ast(self.asts[0]) # length: 206
        self.assertEqual(len(padded_result), 420)
        self.assertEqual(padded_result[208], 0)
        self.assertEqual(padded_result[416], 0)
        self.assertNotEqual(padded_result[210], 0)
        too_long_result = transformer.transform_ast(self.asts[2]) # length: 215
        self.assertFalse(too_long_result)
