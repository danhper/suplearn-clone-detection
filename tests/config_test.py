from tests.base import TestCase

from suplearn_clone_detection.config import Config

class ConfigTest(TestCase):
    @classmethod
    def load_config(cls):
        return Config.from_file(cls.fixture_path("config.yml"))

    @classmethod
    def setUpClass(cls):
        cls.config = cls.load_config()

    def test_config(self):
        config = self.config.model
        self.assertEqual(config.dense_layers, [64, 64])

    def test_hash(self):
        self.assertEqual(self.config.data_generation_checksum(), self.load_config().data_generation_checksum())
