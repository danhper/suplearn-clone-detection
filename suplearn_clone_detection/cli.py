import argparse

from suplearn_clone_detection import trainer


def create_parser():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "-c", "--config", help="config file to train model", default="config.yml")
    train_parser.add_argument(
        "-q", "--quiet", help="reduce output", default=False, action="store_true")

    return parser


def run():
    parser = create_parser()
    args = parser.parse_args()
    if not args.command:
        parser.error("no command provided")
    elif args.command == "train":
        trainer.train(args.config, args.quiet)
