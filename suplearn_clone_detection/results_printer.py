import yaml
import matplotlib.pyplot as plt


METRIC_NAMES = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "avg_precision": "Avg. Precision",
    "pr_curve": "Precision-Recall curve",
    "f1": "F1 score",
}


class ResultsPrinter:
    def __init__(self, results_path: str):
        with open(results_path) as f:
            self.results = yaml.load(f)

    def show(self, metric: str, output: str):
        method = getattr(self, "show_{0}".format(metric), None)
        if method:
            method(output)
        elif metric in self.results:
            print(self.results[metric])
        else:
            raise ValueError("unknown metric {0}".format(metric))

    def show_pr_curve(self, output: str):
        pr_curve = self.results["pr_curve"]
        precision, recall = pr_curve["precision"], pr_curve["recall"]

        plt.step(recall, precision, color="b", alpha=0.2, where="post")
        plt.fill_between(recall, precision, step="post", alpha=0.2, color="b")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title("Precision-Recall curve: AP={0:0.2f}".format(self.results["avg_precision"]))

        if output:
            plt.savefig(output)
        else:
            plt.show()

    def show_summary(self, output: str):
        results = self.results.copy()
        del results["pr_curve"]
        formatted_results = self.format_table(results)
        if output:
            with open(output, "w") as f:
                f.write(formatted_results)
        else:
            print(formatted_results, end="")

    @staticmethod
    def format_table(dict_object):
        key_width = max(len(k) for k in dict_object.keys()) + 1
        value_width = len("Value")

        separator = "+-{0}-+-{1}-+\n".format("-" * key_width, "-" * value_width)
        header = "| {0:>{key_width}} | {1:>{value_width}} |\n".format(
            "Metric", "Value", key_width=key_width, value_width=value_width)
        value_tpl = "| {0:>{key_width}} | {1:<{value_width}.2f} |\n"

        s = separator + header + separator
        for key, value in dict_object.items():
            s += value_tpl.format(METRIC_NAMES[key], value,
                                  key_width=key_width, value_width=value_width)
        s += separator
        return s
