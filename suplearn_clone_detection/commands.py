from typing import Dict
from os import path

from suplearn_clone_detection.evaluator import Evaluator
from suplearn_clone_detection.trainer import Trainer


def train(config_path: str, quiet: bool = False):
    trainer = Trainer(config_path)
    if not quiet:
        print("initializing trainer...")
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


def evaluate(options: Dict[str, str]):
    options = options.copy()

    if options.get("output", "") is None:
        options["output"] = "results-{0}.yml".format(options["data_type"])

    if options.get("base_dir"):
        for key in ["config", "model", "output"]:
            if options.get(key):
                options[key] = path.join(options["base_dir"], options[key])

    for key in ["config", "model"]:
        if not path.isfile(options[key]):
            raise ValueError("cannot open {0}".format(options[key]))

    evaluator = Evaluator.from_config(options["config"], options["model"])
    results = evaluator.evaluate(data_type=options["data_type"],
                                 output=options["output"],
                                 overwrite=options.get("overwrite", False))
    if not options.get("quiet", False):
        evaluator.output_results(results)
    return results
