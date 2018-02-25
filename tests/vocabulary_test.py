import tempfile
from tests.base import TestCase

from suplearn_clone_detection.vocabulary import Vocabulary



class VocabularyTest(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vocab_no_values = Vocabulary.from_file(cls.fixture_path("vocab-noid.tsv"))
        cls.vocab_with_values = Vocabulary.from_file(
            cls.fixture_path("vocab-100.tsv"), fallback_empty_value=False)

    def test_valid_access_no_value(self):
        self.assertEqual(self.vocab_no_values.index({"type": "SimpleName"}), 0)
        self.assertEqual(self.vocab_no_values.index({"type": "BinaryExpr"}), 10)
        self.assertEqual(self.vocab_no_values.index({"type": "DoStmt"}), 60)

    def test_valid_access_with_value(self):
        self.assertEqual(self.vocab_with_values.index({"type": "SimpleName"}), 0)
        self.assertEqual(
            self.vocab_with_values.index({"type": "IntegerLiteralExpr", "value": "0"}), 21)
        self.assertEqual(
            self.vocab_with_values.index({"type": "BooleanLiteralExpr", "value": "true"}), 67)

    def test_keyerror_no_value(self):
        with self.assertRaises(KeyError):
            _ = self.vocab_no_values.index({"type": "IDontExist"})

    def test_keyerror_with_value(self):
        with self.assertRaises(KeyError):
            _ = self.vocab_with_values.index({"type": "IDontExist"})

        with self.assertRaises(KeyError):
            _ = self.vocab_with_values.index({"type": "SimpleName", "value": "dont-exist"})

    def test_save(self):
        with tempfile.NamedTemporaryFile(prefix="suplearn-cc") as f:
            self.vocab_with_values.save(f.name)
            reloaded_vocab = Vocabulary.from_file(f.name, fallback_empty_value=False)
            self.assertEqual(self.vocab_with_values, reloaded_vocab)
