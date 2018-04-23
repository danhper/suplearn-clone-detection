import sys
import argparse
import logging

from suplearn_clone_detection import commands
from suplearn_clone_detection.token_based import commands as token_commands


def create_token_parser(base_subparsers):
    token_parser = base_subparsers.add_parser("tokens", help="Tokens-based data creation commands")
    subparsers = token_parser.add_subparsers(dest="subcommand")

    create_vocab_parser = subparsers.add_parser(
        "create-vocabulary", help="Create vocabulary from tokens file")
    create_vocab_parser.add_argument("input", help="Input file containing tokens")
    create_vocab_parser.add_argument(
        "-o", "--output", help="Vocabulary output file", required=True)
    create_vocab_parser.add_argument(
        "--size", help="Maxiumum size of the vocabulary", type=int, default=10000)
    create_vocab_parser.add_argument(
        "--strip-values",
        help="Maxiumum size of the vocabulary",
        default=True,
        action="store_false",
        dest="include_values")

    create_skipgram_data_parser = subparsers.add_parser(
        "skipgram-data", help="Create data to train skipgram model")
    create_skipgram_data_parser.add_argument("input", help="Input file containing tokens")
    create_skipgram_data_parser.add_argument(
        "-o", "--output", help="Vocabulary output file", required=True)
    create_skipgram_data_parser.add_argument(
        "-v", "--vocabulary", help="Path to the vocabulary file", required=True)
    create_skipgram_data_parser.add_argument(
        "-w", "--window-size", help="Window size to generate context", type=int, default=2)


def make_file_processor_parser(parser, output_required=False):
    parser.add_argument("file", help="file containing list of files to predict")
    parser.add_argument(
        "-b", "--files-base-dir", help="base directory for files in <file>")
    parser.add_argument(
        "-d", "--base-dir", help="base directory for model, config and output")
    parser.add_argument(
        "-c", "--config", help="config file for the model to evaluate", default="config.yml")
    parser.add_argument(
        "-m", "--model", help="path to the model to evaluate", default="model.h5")
    parser.add_argument(
        "--files-cache", help="file containing cached vectors for files")
    parser.add_argument(
        "--asts-path", help="file containing the JSON representation of the ASTs")
    parser.add_argument(
        "--filenames-path", help="file containing the filename path of the ASTs")
    parser.add_argument(
        "--batch-size", help="size of a batch", type=int)
    parser.add_argument(
        "-o", "--output",
        help="file where to save the output", required=output_required)


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


    generate_data_parser = subparsers.add_parser(
        "generate-data", help="Generate data for evaluating model")
    generate_data_parser.add_argument(
        "-c", "--config", help="config file path", default="config.yml")
    generate_data_parser.add_argument(
        "-o", "--output", help="output to save generated files")
    generate_data_parser.add_argument(
        "--data-type", choices=["dev", "test"], default="dev",
        help="the type of data to generate")

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate the model")
    evaluate_parser.add_argument(
        "-d", "--base-dir", help="base directory for model, config and output")
    evaluate_parser.add_argument(
        "--data-path", help="path of the data to use for evaulation (csv file)")
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
    make_file_processor_parser(predict_parser)
    predict_parser.add_argument(
        "--max-size-diff", help="max size diff as a ratio between clones", type=float)

    vectorize_parser = subparsers.add_parser("vectorize", help="Vectorize files")
    make_file_processor_parser(vectorize_parser, output_required=True)

    show_results_parser = subparsers.add_parser("show-results", help="Show formatted results")
    show_results_parser.add_argument("filepath", help="file containing the results")
    show_results_parser.add_argument("metric", help="the metric to show")
    show_results_parser.add_argument("-o", "--output", help="output file to save the result")

    create_token_parser(subparsers)


    return parser


app_parser = create_parser()


def run_token_command(args):
    if args.subcommand == "create-vocabulary":
        token_commands.create_vocabulary(
            args.input, args.size, args.include_values, args.output)
    elif args.subcommand == "skipgram-data":
        token_commands.generate_skipgram_data(
            args.input, args.vocabulary, args.window_size, args.output)
    else:
        app_parser.error("no subcommand provided")


def run_command(args):
    if args.command == "train":
        commands.train(args.config, args.quiet)
    elif args.command == "evaluate":
        commands.evaluate(vars(args))
    elif args.command == "predict":
        commands.predict(vars(args))
    elif args.command == "generate-data":
        commands.generate_data(args.config, args.output, args.data_type)
    elif args.command == "show-results":
        commands.show_results(args.filepath, args.metric, args.output)
    elif args.command == "tokens":
        run_token_command(args)
    elif args.command == "vectorize":
        commands.vectorize(vars(args))
    else:
        app_parser.error("no command provided")


def run():
    args = app_parser.parse_args()

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
