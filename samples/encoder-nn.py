#%%
from os import path

import h5py
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import load_model

from suplearn_clone_detection import layers, util
from suplearn_clone_detection.predictor import Predictor

#%%
PROJECT_DIR = path.expanduser("~/Documents/organizations/tuvistavie/suplearn-clone-detection")
ROOT_DIR = path.join(PROJECT_DIR, "tmp/20180425-2227")
CONFIG_PATH = path.join(ROOT_DIR, "config.yml")
MODEL_PATH = path.join(ROOT_DIR, "model.h5")
EXAMPLES_PATH = path.join(PROJECT_DIR, "tmp/sample-data/java-java-dev.csv")

#%%
model = load_model(MODEL_PATH, custom_objects=layers.custom_objects) # type: layers.ModelWrapper

left_encoder = model.inner_models[0]
right_encoder = model.inner_models[1]
distance_model = model.inner_models[2]
print("MODEL LOADED")

#%%

left_encoder.summary()
right_encoder.summary()
distance_model.summary()

#%%

with open(EXAMPLES_PATH) as f:
    files = []
    for line in f:
        pair = line.strip().split("," if "," in line else " ")[:2]
        files.append(tuple(path.join(filename) for filename in pair))

print("\n".join([" ".join(v) for v in files[:10]]))

#%%
predictor = Predictor.from_config(CONFIG_PATH, MODEL_PATH, {})

#%%
to_predict = files[8:20]
to_predict, input_data, _assumed_false = predictor._generate_vectors(to_predict)
result = model.predict(input_data)
result

#%%

sess = K.get_session()

left = left_encoder(tf.constant(input_data[0]))
right = right_encoder(tf.constant(input_data[1]))
sess.run(distance_model([left, right]))

#%%

def read_inputs(filename, files):
    with h5py.File(path.join(ROOT_DIR, filename)) as f:
        return np.array([f[v][:] for v in files])

left_inputs = read_inputs("java-dev-files-left.h5", [v[0] for v in to_predict])
right_inputs = read_inputs("java-dev-files-right.h5", [v[1] for v in to_predict])

print(left_inputs[0], sess.run(left)[0])
print(right_inputs[0], sess.run(right)[0])

#%%

def find_nearest_neighbors(input_array, files, batch_size=512):
    left_input = tf.reshape(
        tf.tile(tf.constant(input_array), [batch_size]),
        (batch_size, -1))

    results = []
    def run_batch(batch):
        filenames = [v[0] for v in batch]
        right_input = tf.constant(np.array([v[1][:] for v in batch]))
        res = distance_model([left_input[:len(batch)], right_input])
        distances = sess.run(res)
        results.extend(zip(filenames, distances))

    batch = []
    def visitor(name, value):
        if not isinstance(value, h5py.Dataset):
            return
        batch.append((name, value))
        if len(batch) >= batch_size:
            run_batch(batch)
            batch.clear()

    files.visititems(visitor)
    if batch:
        run_batch(batch)

    results = np.array(results, dtype=[("filename", "S256"), ("distance", float)])
    return np.sort(results, order="distance")[::-1]

with h5py.File(path.join(ROOT_DIR, "java-dev-files-right.h5")) as f:
    result = find_nearest_neighbors(left_inputs[0], f)

print("results for {0}".format(to_predict[0][0]))
print(result[:10])


# sess.run(tf.reshape(tf.tile(tf.constant(left_inputs[0]), [5]), (5, -1)))
