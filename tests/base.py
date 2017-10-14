from unittest import TestCase as BaseTestCase

from os import path


FIXTURES_PATH = path.join(path.dirname(__file__), "fixtures")


class TestCase(BaseTestCase):
    @staticmethod
    def fixture_path(filename):
        return path.join(FIXTURES_PATH, filename)
