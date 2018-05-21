from suplearn_clone_detection.data_generator import DataGenerator
from suplearn_clone_detection.ast_transformer import ASTTransformer
from suplearn_clone_detection.config import GeneratorConfig


from tests.base import TestCase


class NoopASTTransformer(ASTTransformer):
    def __init__(self, lang):
        super(NoopASTTransformer, self).__init__(lang, {})

    def transform_ast(self, list_ast):
        return list_ast


class DataGeneratorTest(TestCase):
    @classmethod
    def setUpClass(cls):
        transformers = [NoopASTTransformer("java"), NoopASTTransformer("python")]
        config = GeneratorConfig(dict(submissions_path=cls.fixture_path("submissions.json"),
                                      asts_path=cls.fixture_path("asts.json")))
        cls.generator = DataGenerator(config, transformers)
        cls.iterator = cls.generator.make_iterator()

    def setUp(self):
        self.iterator.reset()

    def test_load_submissions(self):
        self.assertEqual(len(self.generator.submissions), 5)

    def test_group_by_language(self):
        self.assertEqual(len(self.generator.submissions_by_language["java"]), 3)
        self.assertEqual(len(self.generator.submissions_by_language["python"]), 2)
        get_ast_len = lambda sub: len(self.generator.get_ast(sub))
        for submisisons in self.generator.submissions_by_language.values():
            for i in range(1, len(submisisons)):
                self.assertGreaterEqual(get_ast_len(submisisons[i]),
                                        get_ast_len(submisisons[i - 1]))

    def test_group_by_problem(self):
        self.assertEqual(set([("r", 1, 1), ("r", 1, 0), ("b", 5, 0)]),
                         self.generator.submissions_by_problem.keys())
        self.assertEqual(len(self.generator.submissions_by_problem[("r", 1, 1)]), 3)
        self.assertEqual(len(self.generator.submissions_by_problem[("r", 1, 0)]), 1)
        self.assertEqual(len(self.generator.submissions_by_problem[("b", 5, 0)]), 1)

    def test_len(self):
        self.assertEqual(len(self.iterator), 4)

    def test_next_batch(self):
        [lang1_inputs, lang2_inputs], labels, weights = self.iterator.next_batch(4)
        self.assertEqual(len(lang1_inputs), 4)
        self.assertEqual(len(lang2_inputs), 4)
        self.assertEqual(len(labels), 4)
        self.assertEqual(len(weights), 4)
        for label in labels:
            self.assertIn(label, [[0], [1]])

        [lang1_inputs, lang2_inputs], labels, _weights = self.iterator.next_batch(4)
        self.assertEqual(len(lang1_inputs), 0)
        self.assertEqual(len(lang2_inputs), 0)
        self.assertEqual(len(labels), 0)

    def test_reset(self):
        [lang1_inputs, lang2_inputs], _labels, _weights = self.iterator.next_batch(4)
        self.assertEqual(len(lang1_inputs), 4)
        self.assertEqual(len(lang2_inputs), 4)
        [lang1_inputs, _lang2_inputs], _labels, _weights = self.iterator.next_batch(4)
        self.assertEqual(len(lang1_inputs), 0)
        self.iterator.reset()
        [lang1_inputs, _lang2_inputs], _labels, _weights = self.iterator.next_batch(4)
        self.assertEqual(len(lang1_inputs), 4)

    def test_same_language(self):
        transformers = [NoopASTTransformer("java"), NoopASTTransformer("java")]
        config = GeneratorConfig(dict(submissions_path=self.fixture_path("submissions.json"),
                                      asts_path=self.fixture_path("asts.json")))
        generator = DataGenerator(config, transformers)
        iterator = generator.make_iterator()
        [lang1_inputs, lang2_inputs], _labels, _weights = iterator.next_batch(2)
        self.assertEqual(len(lang1_inputs), 2)
        self.assertEqual(len(lang2_inputs), 2)
        [lang1_inputs, _lang2_inputs], _labels, _weights = iterator.next_batch(2)
        self.assertEqual(len(lang1_inputs), 0)
