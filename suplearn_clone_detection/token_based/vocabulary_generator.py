from typing import Tuple, Optional

import json
from suplearn_clone_detection.vocabulary import Vocabulary


def make_key(token: Tuple[str, Optional[str]], include_values: bool):
    token_type, value = token
    key = (token_type,)
    if include_values:
        key += (value,)
    return key


def make_token(index: int, token: Tuple[str, Optional[str]], count: int):
    token_type, value = token
    meta_type = token_type.split(".")[0]
    return dict(
        id=index,
        type=token_type,
        metaType=meta_type,
        value=value,
        count=count
    )


def generate_vocabulary(filepath: str, size: int, include_values: bool) -> Vocabulary:
    counts = {}
    with open(filepath) as f:
        for row in f:
            tokens = json.loads(row)
            for token in tokens:
                key = (token["type"], token.get("value"))
                counts[key] = counts.get(key, 0) + 1
                # allow to fallback to type only
                if include_values and token.get("value") is not None:
                    no_value_key = (token["type"], None)
                    counts[no_value_key] = counts.get(no_value_key, 0) + 1
    sorted_counts = sorted(counts.items(), key=lambda v: -v[1])
    entries = {make_key(token, include_values): make_token(i, token, count)
               for i, (token, count) in enumerate(sorted_counts[:size])}
    return Vocabulary(entries=entries, has_values=include_values)
