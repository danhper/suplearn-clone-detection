import json
import logging
import gzip
from typing import Tuple, List

from suplearn_clone_detection.vocabulary import Vocabulary
from suplearn_clone_detection.token_based import util


class SkipgramGenerator:
    def __init__(self, tokens_path: str, vocabulary_path: str) -> None:
        self.tokens_path = tokens_path
        self.vocabulary = Vocabulary.from_file(vocabulary_path)

    def generate_skipgram_data(self, window_size: int, output: str) -> None:
        total_lines = util.get_lines_count(self.tokens_path)
        logging.info("generating skipgram data from %s files", total_lines)
        with util.open_file(self.tokens_path) as token_files, \
             gzip.open(output, "wb") as output_file:
            for i, row in enumerate(token_files):
                tokens = json.loads(row)
                if isinstance(tokens, dict) and "tokens" in tokens:
                    tokens = tokens["tokens"]
                for target, context in self._generate_context_pairs(tokens, window_size):
                    output_file.write("{0},{1}".format(target, context).encode("utf-8"))
                    output_file.write(b"\n")

                if i > 0 and i % 1000 == 0:
                    logging.info("progress: %s/%s", i, total_lines)

    def _generate_context_pairs(self, tokens: List[dict], window_size: int) \
            -> List[Tuple[int, int]]:
        for i, target in enumerate(tokens):
            target_index = self.vocabulary.index(target)
            for j in range(max(0, i - window_size), min(i + window_size + 1, len(tokens))):
                if i != j:
                    context_index = self.vocabulary.index(tokens[j])
                    yield (target_index, context_index)
