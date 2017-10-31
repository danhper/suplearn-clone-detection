import csv
import numpy as np


class Vocabulary:
    def __init__(self, filepath):
        self.vocabulary, self.has_values = self._parse_file(filepath)

    @staticmethod
    def _parse_file(filepath):
        with open(filepath, "r", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            has_values = "value" in reader.fieldnames
            vocabulary = {}
            for row in reader:
                key = (row["type"],)
                if has_values:
                    key += (row.get("value"),)
                letter_id = np.int32(int(row["id"]))
                vocabulary[key] = letter_id
            return vocabulary, has_values

    def __len__(self):
        return len(self.vocabulary)

    def __getitem__(self, key):
        if not self.has_values:
            return self.vocabulary[(key["type"],)]
        return self.vocabulary[(key["type"], key.get("value"))]
