import csv
import json
import pandas as pd
import numpy as np


class Vocabulary:
    def __init__(self, filepath):
        self.vocabulary = pd.read_csv(filepath, sep="\t", quoting=csv.QUOTE_NONE)
        self.has_values = "value" in self.vocabulary
        if self.has_values:
            self.vocabulary.value = self.vocabulary.value.apply(self._transform_value)
        self.vocabulary.set_index(self.indexes, inplace=True, verify_integrity=True)
        self.vocabulary.sort_index(level=self.indexes, inplace=True)

    @staticmethod
    def _transform_value(value):
        if isinstance(value, str):
            return json.loads(value)
        return value

    @property
    def indexes(self):
        if self.has_values:
            return ["type", "value"]
        return "type"

    def __getitem__(self, key):
        if not self.has_values:
            return self.vocabulary.loc[key["type"]].id
        return self.vocabulary.loc[key["type"], key.get("value", np.nan)].id
