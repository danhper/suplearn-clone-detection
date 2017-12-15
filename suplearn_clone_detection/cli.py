import sys
import argparse
import logging

from suplearn_clone_detection import commands


def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-q", "--quiet", help="reduce output", default=False, action="store_true")
    parser.add_argument(
        "--debug", help="enables debug", default=False, action="store_true")

    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "-c", "--config", help="config file to train model", default="config.yml")

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate the model")
    evaluate_parser.add_argument(
        "-d", "--base-dir", help="base directory for model, config and output")
    evaluate_parser.add_argument(
        "--data-type", choices=["dev", "test"], default="dev",
        help="the type of data on which to evaluate the model")
    evaluate_parser.add_argument(
        "-c", "--config", help="config file for the model to evaluate", default="config.yml")
    evaluate_parser.add_argument(
        "-m", "--model", help="path to the model to evaluate", default="model.h5")
    evaluate_parser.add_argument(
        "-o", "--output", help="file where to save the output")
    evaluate_parser.add_argument(
        "-f", "--overwrite", help="overwrite the results output if file exists",
        default=False, action="store_true")


    predict_parser = subparsers.add_parser("predict", help="Predict files")
    predict_parser.add_argument("file", help="file containing list of files to predict")
    predict_parser.add_argument(
        "-b", "--files-base-dir", help="base directory for files in <file>")
    predict_parser.add_argument(
        "-d", "--base-dir", help="base directory for model, config and output")
    predict_parser.add_argument(
        "-c", "--config", help="config file for the model to evaluate", default="config.yml")
    predict_parser.add_argument(
        "-m", "--model", help="path to the model to evaluate", default="model.h5")
    predict_parser.add_argument(
        "--files-cache", help="file containing cached vectors for files")
    predict_parser.add_argument(
        "--asts-path", help="file containing the JSON representation of the ASTs")
    predict_parser.add_argument(
        "--filenames-path", help="file containing the filename path of the ASTs")
    predict_parser.add_argument(
        "--batch-size", help="size of a batch", type=int)
    predict_parser.add_argument(
        "-o", "--output", help="file where to save the output")

    return parser


def run_command(args):
    if args.command == "train":
        commands.train(args.config, args.quiet)
    elif args.command == "evaluate":
        commands.evaluate(vars(args))
    elif args.command == "predict":
        commands.predict(vars(args))


def run():
    parser = create_parser()
    args = parser.parse_args()
    if not args.command:
        parser.error("no command provided")

    log_level = logging.INFO if args.quiet else logging.DEBUG
    logging.basicConfig(level=log_level,
                        format="%(asctime)-15s %(levelname)s %(message)s")

    if args.debug:
        return run_command(args)

    try:
        run_command(args)
    except Exception as e: # pylint: disable=broad-except
        logging.error("failed: %s", e)
        sys.exit(1)
