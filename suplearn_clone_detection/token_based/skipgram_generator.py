import json
import gzip
from typing import Tuple, List

from suplearn_clone_detection.vocabulary import Vocabulary

class SkipgramGenerator:
    def __init__(self, tokens_path: str, vocabulary_path: str) -> None:
        self.tokens_path = tokens_path
        self.vocabulary = Vocabulary.from_file(vocabulary_path)

    def generate_skipgram_data(self, window_size: int, output: str) -> None:
        with open(self.tokens_path) as token_files, \
             gzip.open(output, "wb") as output_file:
            for row in token_files:
                tokens = json.loads(row)
                for target, context in self.generate_context_pairs(tokens, window_size):
                    print("{0},{1}".format(target, context), file=output_file)

    def generate_context_pairs(self, tokens: List[dict], window_size: int) \
            -> List[Tuple[int, int]]:
        for i, target in enumerate(tokens):
            target_index = self.vocabulary.index(target)
            for j in range(max(0, i - window_size), min(i + window_size + 1, len(tokens))):
                if i != j:
                    context_index = self.vocabulary.index(tokens[j])
                    yield (target_index, context_index)
