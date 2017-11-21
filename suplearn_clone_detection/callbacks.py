from keras.callbacks import Callback

from suplearn_clone_detection.evaluator import Evaluator


class EvaluateModelCallback(Callback):
    def __init__(self, data_generator, output=None, quiet=False):
        super(EvaluateModelCallback, self).__init__()
        self.data_generator = data_generator
        self.output = output
        self.quiet = quiet

    def on_epoch_end(self, epoch, logs=None):
        evaluator = Evaluator(self.model, self.data_generator)
        output = self.output.format(epoch=epoch) if self.output else None
        results = evaluator.evaluate(data_type="dev", output=output)
        if not self.quiet:
            print("\nDev set results")
            evaluator.output_results(results)
