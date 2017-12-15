from tests.base import TestCase

from suplearn_clone_detection.ast_loader import ASTLoader

class ASTLoaderTest(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loader = ASTLoader(cls.fixture_path("asts.json"))

    def test_load_names(self):
        self.assertEqual(len(self.loader.names), 5)

    def test_load_asts(self):
        self.assertEqual(len(self.loader.asts), 5)
