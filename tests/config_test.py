import configparser
from tests.base import TestCase
import yaml

from suplearn_clone_detection.config import ModelConfig

class ConfigTest(TestCase):
    @classmethod
    def setUpClass(cls):
        with open(cls.fixture_path("config.yml")) as f:
            cls.config = yaml.load(f)

    def test_config(self):
        config = ModelConfig(self.config["model"])
        self.assertEqual(config.dense_layers, [64, 64])
