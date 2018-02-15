import csv
import json
import numpy as np


class Vocabulary:
    @classmethod
    def from_file(cls, filepath, fallback_empty_value=True):
        entries, headers, has_values = cls._parse_file(filepath)
        vocab = Vocabulary(
            headers=headers,
            has_values=has_values,
            fallback_empty_value=fallback_empty_value)
        vocab.entries = entries
        return vocab

    def __init__(self, headers, has_values=True, fallback_empty_value=True):
        self.headers = headers
        self.has_values = has_values
        self.fallback_empty_value = fallback_empty_value
        self.entries = {}

    def __eq__(self, other):
        if not isinstance(other, Vocabulary):
            return False
        attrs = ["headers", "entries", "has_values", "fallback_empty_value"]
        return all(getattr(self, attr) == getattr(other, attr) for attr in attrs)

    @staticmethod
    def _parse_file(filepath):
        entries = {}
        with open(filepath, "r", newline="") as f:
            # reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            headers = next(f).strip().split("\t")
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
                letter_id = np.int32(entry["id"])
                entries[key] = {"id": letter_id, "data": entry}
        return entries, headers, has_values

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, token):
        if not self.has_values:
            return self.entries[(token["type"],)]["id"]

        key = (token["type"], token.get("value"))
        result = self.entries.get(key)
        if result is not None:
            return result["id"]
        elif self.fallback_empty_value:
            return self.entries[(token["type"], None)]["id"]
        else:
            raise KeyError(key)

    def save(self, path, offset=0):
        with open(path, "w") as f:
            print("\t".join(self.headers), file=f)
            for i in range(offset):
                row = [str(i), "offset-{0}".format(i), "padding", "0"]
                if self.has_values:
                    row.append('"padding"')
                print("\t".join(row), file=f)
            for entry in sorted(self.entries.values(), key=lambda x: x["id"]):
                data = entry["data"]
                row = [str(data["id"] + offset), data["type"],
                       data["metaType"], str(data["count"])]
                if self.has_values:
                    row.append(json.dumps(data["value"]) if data["value"] else "")
                print("\t".join(row), file=f)
