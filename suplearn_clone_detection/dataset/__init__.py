from keras.utils import Sequence

from suplearn_clone_detection.config import Config
from suplearn_clone_detection.dataset.sequences import DevSequence, TestSequence


def get(config: Config, data_type: str) -> Sequence:
    if data_type == "dev":
        return DevSequence(config)
    elif data_type == "test":
        return TestSequence(config)
    raise ValueError("cannot get {0} sequence".format(data_type))
