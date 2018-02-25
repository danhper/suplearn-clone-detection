import json
import numpy as np


BASE_HEADERS = ["id", "type", "metaType", "count"]


class Vocabulary:
    @classmethod
    def from_file(cls, filepath, fallback_empty_value=True):
        entries, has_values = cls._parse_file(filepath)
        vocab = Vocabulary(
            entries=entries,
            has_values=has_values,
            fallback_empty_value=fallback_empty_value)
        return vocab

    def __init__(self, entries=None, has_values=True, fallback_empty_value=True):
        self.headers = BASE_HEADERS.copy()
        if has_values:
            self.headers.append("value")
        self.has_values = has_values
        self.fallback_empty_value = fallback_empty_value
        if not entries:
            entries = {}
        self.entries = entries

    def __eq__(self, other):
        if not isinstance(other, Vocabulary):
            return False
        attrs = ["headers", "entries", "has_values", "fallback_empty_value"]
        return all(getattr(self, attr) == getattr(other, attr) for attr in attrs)

    @staticmethod
    def _parse_file(filepath):
        entries = {}
        with open(filepath, "r", newline="") as f:
            headers = next(f).strip().split("\t")
            assert BASE_HEADERS == headers[:len(BASE_HEADERS)]
            has_values = "value" in headers
            entries = {}
            for row in f:
                row = row.strip().split("\t")
                entry = dict(id=int(row[0]), type=row[1], metaType=row[2], count=int(row[3]))
                if has_values:
                    entry["value"] = json.loads(row[4]) if len(row) > 4 and row[4] else None
                key = (entry["type"],)
                if has_values:
                    key += (entry["value"],)
                entry["id"] = int(entry["id"])
                entries[key] = entry
        return entries, has_values

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, token):
        if isinstance(token, tuple):
            return self.entries[token]

        if not self.has_values:
            return self.entries[(token["type"],)]

        key = (token["type"], token.get("value"))
        result = self.entries.get(key)
        if result is not None:
            return result
        elif self.fallback_empty_value:
            return self.entries[(token["type"], None)]
        else:
            raise KeyError(key)

    def index(self, token):
        return np.int32(self[token]["id"])

    def save(self, path, offset=0):
        with open(path, "w") as f:
            print("\t".join(self.headers), file=f)
            for i in range(offset):
                row = [str(i), "offset-{0}".format(i), "padding", "0"]
                if self.has_values:
                    row.append('"padding"')
                print("\t".join(row), file=f)
            for entry in sorted(self.entries.values(), key=lambda x: x["id"]):
                row = [str(entry["id"] + offset), entry["type"],
                       entry.get("metaType", "Other"), str(entry["count"])]
                if self.has_values:
                    row.append(json.dumps(entry["value"]) if entry["value"] else "")
                print("\t".join(row), file=f)
