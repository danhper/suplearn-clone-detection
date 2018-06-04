import csv
from typing import Dict
from os import path
import logging

from suplearn_clone_detection import ast_transformer, database
from suplearn_clone_detection.config import Config
from suplearn_clone_detection.data_generator import DataGenerator
from suplearn_clone_detection.dataset.generator import DatasetGenerator
from suplearn_clone_detection.evaluator import Evaluator
from suplearn_clone_detection.predictor import Predictor
from suplearn_clone_detection.vectorizer import Vectorizer
from suplearn_clone_detection.results_printer import ResultsPrinter
from suplearn_clone_detection.trainer import Trainer


def train(config_path: str, quiet: bool = False):
    trainer = Trainer(config_path, quiet)
    logging.debug("initializing trainer...")
    trainer.initialize()
    if not quiet:
        trainer.model.summary()
    trainer.train()

    evaluator = Evaluator.from_trainer(trainer)
    results_file = path.join(trainer.output_dir, "results-dev.yml")
    results = evaluator.evaluate(output=results_file)
    if not quiet:
        evaluator.output_results(results)
    return results


def generate_data(config_path: str, output: str, data_type: str = "dev"):
    config = load_and_process_config(config_path)
    transformers = ast_transformer.create_all(config.model.languages)
    data_generator = DataGenerator(config.generator, transformers)
    data_it = data_generator.make_iterator(data_type=data_type)
    with open(output, "w") as f:
        logging.info("outputing %s samples into %s", len(data_it), output)
        csvwriter = csv.writer(f)
        for x, y in data_it.iterate():
            left, right = [v.submission["file"] for v in x]
            csvwriter.writerow([left, right, str(y)])


def evaluate(options: Dict[str, str]):
    options = process_options(options)
    config = load_and_process_config(options["config"])

    if options.get("output", "") is None:
        val = "results-{0}.yml".format(options["data_type"])
        options["output"] = path.join(options.get("base_dir", ""), val)

    evaluator = Evaluator.from_config(config, options["model"])
    results = evaluator.evaluate(data_path=options["data_path"],
                                 data_type=options["data_type"],
                                 output=options["output"],
                                 overwrite=options.get("overwrite", False))
    if not options.get("quiet", False):
        evaluator.output_results(results)
    return results


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
