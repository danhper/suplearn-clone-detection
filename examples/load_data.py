from os import path

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
    generator = DataGenerator(SUBMISSIONS_PATH, ASTS_PATH, transformer)
    print("finished loading data")

    print("first batch")
    generator.next_batch(128)
    print("second batch")
    generator.next_batch(128)


if __name__ == '__main__':
    main()
