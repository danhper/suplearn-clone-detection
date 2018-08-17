import math
from keras.models import Model
import h5py
import numpy as np
from tqdm import tqdm


from suplearn_clone_detection import util

class Detector:
    def __init__(self, model: Model, dataset: h5py.Dataset):
        self.model = model
        self.dataset = dataset
        self.left_lang = self._get_lang(self.model.layers[0])
        self.right_lang = self._get_lang(self.model.layers[1])

    def _get_lang(self, layer):
        return layer.name.split("_")[1]

    # TODO: parallelize calls to model.predict
    def detect_clones(self, batch_size=1024):
        total_batches = self.batches_count(batch_size)
        for batch in tqdm(self.batch_iterator(batch_size), total=total_batches):
            left, right = self.get_inputs(batch)
            batch_predictions = self.model.predict([left, right]).reshape(len(batch))
            yield from zip(batch, batch_predictions)

    @staticmethod
    def output_prediction_results(predictions, f):
        for (left, right), output in predictions:
            print("{0},{1},{2}".format(left, right, output), file=f)

    def get_inputs(self, batch):
        left, right = zip(*[(self.dataset[l].value, self.dataset[r].value) for l, r in batch])
        return np.array(left), np.array(right)

    def batch_iterator(self, batch_size):
        pairs = util.hdf5_key_pairs(self.dataset, self.left_lang, self.right_lang)
        yield from util.in_batch(pairs, batch_size)

    def batches_count(self, batch_size):
        keys_count = len(util.hdf5_keys(self.dataset))
        pairs_count = (keys_count * (keys_count - 1)) // 2
        return math.ceil(pairs_count / batch_size)
