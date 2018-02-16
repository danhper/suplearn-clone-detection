from os import path
from suplearn_clone_detection.token_based import vocabulary_generator
from suplearn_clone_detection.token_based.skipgram_generator import SkipgramGenerator


def check_path(filepath: str):
    if not path.exists(filepath):
        raise ValueError("{0} does not exist".format(filepath))


def create_vocabulary(filepath: str, size: int, include_values: bool, output: str):
    check_path(filepath)
    vocab = vocabulary_generator.generate_vocabulary(filepath, size, include_values)
    vocab.save(output)


def generate_skipgram_data(filepath: str, vocabulary_path: str, window_size: int, output: str):
    check_path(filepath)
    check_path(vocabulary_path)
    skipgram_generator = SkipgramGenerator(filepath, vocabulary_path)
    skipgram_generator.generate_skipgram_data(window_size, output)
