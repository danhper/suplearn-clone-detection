from functools import total_ordering
from typing import Optional


@total_ordering
class VocabItem:
    def __init__(self, token_type: str, token_value: Optional[str]):
        self.type = token_type
        self.value = token_value
        self.count = 0

    def __hash__(self):
        return hash((self.type, self.value))

    def __eq__(self, other):
        return isinstance(other, VocabItem) and \
               ((self.type, self.value, self.count) == (other.type, other.value, other.count))

    def __lt__(self, other):
        if not isinstance(other, VocabItem):
            return NotImplemented
        if (self.value is None) == (other.value is None):
            return self.count < other.count
        return other.value is None

    def make_key(self, include_values: bool):
        key = (self.type,)
        if include_values:
            key += (self.value,)
        return key

    def make_token(self, index: int):
        meta_type = self.type.split(".")[0]
        return dict(
            id=index,
            type=self.type,
            metaType=meta_type,
            value=self.value,
            count=self.count
        )
