from keras.callbacks import Callback

from suplearn_clone_detection import evaluator


class ModelResultsTracker:
    def __init__(self, data, model, comparator=None):
        self.data = data
        self.model = model
        self._results_cache = {}
        self.best_epoch = -1
        self.best_results = None
        self.comparator = comparator
        self.evaluator = evaluator.Evaluator(self.model)
        if self.comparator is None:
            self.comparator = self.default_comparator

    def compute_results(self, epoch):
        if epoch in self._results_cache:
            return self._results_cache[epoch]
        results = self.evaluator.evaluate(self.data)
        if self.best_results is None or self.comparator(results, self.best_results):
            self.best_epoch = epoch
            self.best_results = results
        self._results_cache[epoch] = results
        return results

    def is_best_epoch(self, epoch):
        if not epoch in self._results_cache:
            self.compute_results(epoch)
        return self.best_epoch == epoch

    @staticmethod
    def default_comparator(current_results, best_results):
        return current_results["f1"] > best_results["f1"]


class ModelEvaluator(Callback):
    def __init__(self, results_tracker, filepath=None, quiet=False, save_best_only=True):
        super(ModelEvaluator, self).__init__()
        self.filepath = filepath
        self.quiet = quiet
        self.save_best_only = save_best_only
        self.results_tracker = results_tracker
        self.best_results = None

    def on_epoch_end(self, epoch, logs=None):
        results = self.results_tracker.compute_results(epoch)
        if not self.quiet:
            print("\nDev set results")
            evaluator.output_results(results)

        if not self.filepath:
            return

        if not self.save_best_only or self.results_tracker.is_best_epoch(epoch):
            filepath = self.filepath.format(epoch=epoch)
            with open(filepath, "w") as f:
                evaluator.output_results(results, file=f)


class ModelCheckpoint(Callback):
    def __init__(self, results_tracker, filepath, save_best_only=False):
        super(ModelCheckpoint, self).__init__()
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.results_tracker = results_tracker
        self.best_results = None

    def on_epoch_end(self, epoch, logs=None):
        if not self.save_best_only or self.results_tracker.is_best_epoch(epoch):
            filepath = self.filepath.format(epoch=epoch)
            self.model.save(filepath, overwrite=True)
