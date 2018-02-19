import logging
from typing import Tuple, Optional, Dict

import json
from suplearn_clone_detection.vocabulary import Vocabulary
from suplearn_clone_detection.token_based import util
from suplearn_clone_detection.token_based.vocab_item import VocabItem


def get_or_add_token(counts: Dict[Tuple[str, Optional[str]], VocabItem],
                     token_type: str, token_value: Optional[str]) -> VocabItem:
    key = (token_type, token_value)
    if key in counts:
        return counts[key]
    item = VocabItem(token_type, token_value)
    counts[key] = item
    return item


def generate_vocabulary(filepath: str, size: int, include_values: bool) -> Vocabulary:
    counts = {}
    total_lines = util.get_lines_count(filepath)
    logging.info("generating vocabulary from %s files", total_lines)
    with util.open_file(filepath) as f:
        for i, row in enumerate(f):
            tokens = json.loads(row)
            if isinstance(tokens, dict) and "tokens" in tokens:
                tokens = tokens["tokens"]
            for token in tokens:
                if include_values:
                    get_or_add_token(counts, token["type"], token.get("value")).count += 1
                if not include_values or token.get("value") is not None:
                    get_or_add_token(counts, token["type"], None).count += 1
            if i > 0 and i % 1000 == 0:
                logging.info("progress: %s/%s", i, total_lines)
    vocab_items = sorted(counts.values(), reverse=True)[:size]
    entries = {item.make_key(include_values): item.make_token(i)
               for i, item in enumerate(vocab_items)}
    return Vocabulary(entries=entries, has_values=include_values)
