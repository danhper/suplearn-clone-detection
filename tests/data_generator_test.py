from suplearn_clone_detection.data_generator import DataGenerator
from suplearn_clone_detection.ast_transformer import ASTTransformer
from suplearn_clone_detection.config import GeneratorConfig


from tests.base import TestCase


class NoopASTTransformer(ASTTransformer):
    def transform_ast(self, ast):
        return ast


class DataGeneratorTest(TestCase):
    @classmethod
    def setUpClass(cls):
        transformers = {"java": NoopASTTransformer(), "python": NoopASTTransformer()}
        config = GeneratorConfig(dict(submissions_path=cls.fixture_path("submissions.json"),
                                      asts_path=cls.fixture_path("asts.json")))
        cls.generator = DataGenerator(config, transformers)

    def setUp(self):
        self.generator.reset()

    def test_load_submissions(self):
        self.assertEqual(len(self.generator.submissions), 5)

    def test_load_names(self):
        self.assertEqual(len(self.generator.names), 5)

    def test_load_asts(self):
        self.assertEqual(len(self.generator.asts), 5)

    def test_group_by_language(self):
        self.assertEqual(len(self.generator.submissions_by_language["java"]), 3)
        self.assertEqual(len(self.generator.submissions_by_language["python"]), 2)

    def test_group_by_problem(self):
        self.assertEqual(set([(1, 1), (1, 0), (5, 0)]),
                         self.generator.submissions_by_problem.keys())
        self.assertEqual(len(self.generator.submissions_by_problem[(1, 1)]), 3)
        self.assertEqual(len(self.generator.submissions_by_problem[(1, 0)]), 1)
        self.assertEqual(len(self.generator.submissions_by_problem[(5, 0)]), 1)

    def test_len(self):
        self.assertEqual(len(self.generator), 4)

    def test_next_batch(self):
        [lang1_inputs, lang2_inputs], labels = self.generator.next_batch(4)
        self.assertEqual(len(lang1_inputs), 4)
        self.assertEqual(len(lang2_inputs), 4)
        self.assertEqual(len(labels), 4)
        for label in labels:
            self.assertIn(label, [[0], [1]])

        [lang1_inputs, lang2_inputs], labels = self.generator.next_batch(4)
        self.assertEqual(len(lang1_inputs), 0)
        self.assertEqual(len(lang2_inputs), 0)
        self.assertEqual(len(labels), 0)

    def test_reset(self):
        [lang1_inputs, lang2_inputs], _labels = self.generator.next_batch(4)
        self.assertEqual(len(lang1_inputs), 4)
        self.assertEqual(len(lang2_inputs), 4)
        [lang1_inputs, _lang2_inputs], _labels = self.generator.next_batch(4)
        self.assertEqual(len(lang1_inputs), 0)
        self.generator.reset()
        [lang1_inputs, _lang2_inputs], _labels = self.generator.next_batch(4)
        self.assertEqual(len(lang1_inputs), 4)
