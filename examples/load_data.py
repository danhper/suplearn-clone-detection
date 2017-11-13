from os import path

import pandas as pd

from suplearn_clone_detection.vocabulary import Vocabulary
from suplearn_clone_detection.data_generator import DataGenerator
from suplearn_clone_detection.ast_transformer import DFSTransformer
from suplearn_clone_detection.config import GeneratorConfig

JAVA_VOCAB_PATH = path.expanduser("~/workspaces/research/results/java/vocabulary/vocab-no-id.tsv")
PYTHON_VOCAB_PATH = path.expanduser("~/workspaces/research/results/python/vocabulary/vocab-no-id.tsv")

SUBMISSIONS_PATH = path.expanduser("~/workspaces/research/dataset/atcoder/submissions.json")
ASTS_PATH = path.expanduser("~/workspaces/research/dataset/atcoder/asts/asts.json")


def main():
    java_vocab = Vocabulary(JAVA_VOCAB_PATH)
    python_vocab = Vocabulary(PYTHON_VOCAB_PATH)

    transformers = {"java": DFSTransformer(java_vocab),
                    "python": DFSTransformer(python_vocab)}

    print("loading data...")
    config = GeneratorConfig(dict(submissions_path=SUBMISSIONS_PATH, asts_path=ASTS_PATH))
    generator = DataGenerator(config, transformers)
    print(len(generator))
    print("finished loading data")

    java_submission_lengths = [len(transformers["java"].transform_ast(generator.get_ast(s)))
                               for s in generator.submissions_by_language["java"]]
    python_submission_lengths = [len(transformers["python"].transform_ast(generator.get_ast(s)))
                                 for s in generator.submissions_by_language["python"]]

    java_lengths = pd.Series(data=java_submission_lengths)
    print(java_lengths.describe())
    print(java_lengths.quantile(0.8))

    python_lengths = pd.Series(data=python_submission_lengths)
    python_lengths.describe()

    python_lengths.quantile(0.95)
    iterator = generator.make_iterator("training")

    print("first batch")
    iterator.next_batch(128)
    print("second batch")
    iterator.next_batch(128)


if __name__ == '__main__':
    main()
