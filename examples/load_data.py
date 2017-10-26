from os import path

import pandas as pd

from suplearn_clone_detection.vocabulary import Vocabulary
from suplearn_clone_detection.data_generator import DataGenerator
from suplearn_clone_detection.ast_transformer import FlatVectorIndexASTTransformer

JAVA_VOCAB_PATH = path.expanduser("~/workspaces/research/results/java/data/no-id.tsv")
PYTHON_VOCAB_PATH = path.expanduser("~/workspaces/research/results/python/vocabulary/no-id.tsv")

SUBMISSIONS_PATH = path.expanduser("~/workspaces/research/dataset/atcoder/submissions.json")
ASTS_PATH = path.expanduser("~/workspaces/research/dataset/atcoder/asts/asts.json")


def main():
    java_vocab = Vocabulary(JAVA_VOCAB_PATH)
    python_vocab = Vocabulary(PYTHON_VOCAB_PATH)

    vocabularies = {"java": java_vocab, "python": python_vocab}

    transformer = FlatVectorIndexASTTransformer(vocabularies)

    print("loading data...")
    generator = DataGenerator(SUBMISSIONS_PATH, ASTS_PATH, transformer, input_max_length=150)
    print(len(generator))
    print("finished loading data")

    java_submission_lengths = [len(transformer.transform_ast(generator.get_ast(s), "java"))
                               for s in generator.submissions_by_language["java"]]
    python_submission_lengths = [len(transformer.transform_ast(generator.get_ast(s), "python"))
                                 for s in generator.submissions_by_language["python"]]

    java_lengths = pd.Series(data=java_submission_lengths)
    print(java_lengths.describe())
    print(java_lengths.quantile(0.8))

    python_lengths = pd.Series(data=python_submission_lengths)
    python_lengths.describe()

    python_lengths.quantile(0.95)

    print("first batch")
    generator.next_batch(128)
    print("second batch")
    generator.next_batch(128)


if __name__ == '__main__':
    main()
