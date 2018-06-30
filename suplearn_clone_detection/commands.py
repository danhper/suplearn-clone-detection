from typing import Dict
from os import path
import logging

import h5py
from keras.models import load_model

from suplearn_clone_detection import database, dataset, layers, evaluator
from suplearn_clone_detection.config import Config
from suplearn_clone_detection.dataset.generator import DatasetGenerator
from suplearn_clone_detection.predictor import Predictor
from suplearn_clone_detection.vectorizer import Vectorizer
from suplearn_clone_detection.results_printer import ResultsPrinter
from suplearn_clone_detection.trainer import Trainer
from suplearn_clone_detection.detector import Detector


def train(config_path: str, quiet: bool = False):
    trainer = Trainer(config_path, quiet)
    logging.debug("initializing trainer...")
    trainer.initialize()
    if not quiet:
        trainer.model.summary()
    trainer.train()

    data = dataset.get(trainer.config, "dev")
    ev = evaluator.Evaluator(trainer.model)
    results_file = path.join(trainer.output_dir, "results-dev.yml")
    results = ev.evaluate(data, output=results_file)
    if not quiet:
        evaluator.output_results(results)
    return results


def evaluate(options: Dict[str, str]):
    options = process_options(options)
    config = load_and_process_config(options["config"])

    if options.get("output", "") is None:
        val = "results-{0}.yml".format(options["data_type"])
        options["output"] = path.join(options.get("base_dir", ""), val)

    ev = evaluator.Evaluator(options["model"])
    data = dataset.get(config, options["data_type"])
    overwrite = options.get("overwrite", False)
    results = ev.evaluate(data, output=options["output"], overwrite=overwrite)
    if not options.get("quiet", False):
        evaluator.output_results(results)
    return results


def evaluate_predictions(options: Dict[str, str]):
    evaluator.evaluate_predictions(options["predictions"], options["output"])


def predict(options: Dict[str, str]):
    options = process_options(options)
    config = load_and_process_config(options["config"])
    with open(options["file"], "r") as f:
        files_base_dir = options.get("files_base_dir") or ""
        files = []
        for line in f:
            pair = line.strip().split("," if "," in line else " ")[:2]
            files.append(tuple(path.join(files_base_dir, filename) for filename in pair))

    predictor = Predictor.from_config(config, options["model"], options)
    predictor.predict(files)

    if not options.get("quiet", False):
        print(predictor.formatted_predictions)

    if options.get("output"):
        with open(options["output"], "w") as f:
            f.write(predictor.formatted_predictions)


def vectorize(options: Dict[str, str]):
    options = process_options(options)
    config = load_and_process_config(options["config"])
    vectorizer = Vectorizer.from_config(config, options["model"], options)
    with open(options["file"]) as f:
        filenames = f.read().splitlines()
    vectorizer.process(filenames, options["output"])


def process_options(options: Dict[str, str]):
    options = options.copy()

    if not options.get("base_dir"):
        return options
    for key in ["config", "model"]:
        if options.get(key):
            options[key] = path.join(options["base_dir"], options[key])

    for key in ["config", "model"]:
        if not path.isfile(options[key]):
            raise ValueError("cannot open {0}".format(options[key]))

    return options


def detect_clones(options: dict):
    model = load_model(options["model"], custom_objects=layers.custom_objects)
    with h5py.File(options["dataset"]) as data:
        detector = Detector(model, data)
        predictions = detector.detect_clones()
        with open(options["output"], "w") as f:
            detector.output_prediction_results(predictions, f)


def show_results(filepath: str, metric: str, output: str):
    printer = ResultsPrinter(filepath)
    printer.show(metric, output)


def generate_dataset(config_path: str):
    config = load_and_process_config(config_path)
    dataset_generator = DatasetGenerator(config)
    dataset_generator.create_samples()


def load_and_process_config(config_path: str) -> Config:
    config = Config.from_file(config_path)
    if config.generator.db_path:
        database.bind_db(config.generator.db_path)
    return config
