import json
import subprocess
from os import path

from keras.models import load_model
import numpy as np

from suplearn_clone_detection.vocabulary import Vocabulary
from suplearn_clone_detection.ast_transformer import FlatVectorIndexASTTransformer


def memoize(f):
    memo = {}
    def wrapper(*args):
        args = tuple(args)
        if args not in memo:
            memo[args] = f(*args)
        return memo[args]
    return wrapper


model = load_model("./tmp/model.h5")

JAVA_VOCAB_PATH = path.expanduser("~/workspaces/research/results/java/vocabulary/vocab-no-id.tsv")
PYTHON_VOCAB_PATH = path.expanduser("~/workspaces/research/results/python/vocabulary/vocab-no-id.tsv")


java_vocab = Vocabulary(JAVA_VOCAB_PATH)
python_vocab = Vocabulary(PYTHON_VOCAB_PATH)

java_transformer = FlatVectorIndexASTTransformer(java_vocab, 1, 200)
python_transformer = FlatVectorIndexASTTransformer(python_vocab, 1, 150)


def get_file_ast(filename):
    _, ext = path.splitext(filename)
    executable = "bigcode-astgen-{0}".format(ext[1:])
    res = subprocess.run([executable, filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        raise ValueError("got exit code {0}: {1}".format(res.returncode, res.stderr))
    return json.loads(res.stdout)


@memoize
def get_file_vector(filename):
    transformer = java_transformer if filename.endswith("java") else python_transformer
    return transformer.transform_ast(get_file_ast(filename))


def load_files(files):
    java_files = []
    python_files = []
    for (java_file, python_file) in files:
        java_files.append(get_file_vector(java_file))
        python_files.append(get_file_vector(python_file))
    return [np.array(java_files), np.array(python_files)]


def predict(files):
    to_predict = load_files(files)
    predictions = [v[0] for v in model.predict(to_predict)]
    return list(zip(files, predictions))



predict([
    ("./tmp/snippets/Fact.java", "./tmp/snippets/fact.py"),
    ("./tmp/snippets/Fibo.java", "./tmp/snippets/fibo.py"),
    ("./tmp/snippets/Quicksort.java", "./tmp/snippets/quicksort.py"),
    ("./tmp/snippets/Hanoi.java", "./tmp/snippets/hanoi.py"),
    ("./tmp/snippets/FizzBuzz.java", "./tmp/snippets/fizzbuzz.py"),
    ("./tmp/snippets/KnuthShuffle.java", "./tmp/snippets/knuth_shuffle.py"),
    ("./tmp/snippets/Cipher.java", "./tmp/snippets/cipher.py"),

    ("./tmp/snippets/Fibo.java", "./tmp/snippets/fact.py"),
    ("./tmp/snippets/Fact.java", "./tmp/snippets/fibo.py"),
    ("./tmp/snippets/Fact.java", "./tmp/snippets/quicksort.py"),
    ("./tmp/snippets/Fact.java", "./tmp/snippets/fizzbuzz.py"),
    ("./tmp/snippets/Quicksort.java", "./tmp/snippets/fibo.py"),
    ("./tmp/snippets/Fibo.java", "./tmp/snippets/quicksort.py"),
    ("./tmp/snippets/Fibo.java", "./tmp/snippets/hanoi.py"),
    ("./tmp/snippets/Hanoi.java", "./tmp/snippets/fibo.py"),
    ("./tmp/snippets/Hanoi.java", "./tmp/snippets/cipher.py"),
    ("./tmp/snippets/Quicksort.java", "./tmp/snippets/hanoi.py"),
    ("./tmp/snippets/Comb.java", "./tmp/snippets/hanoi.py"),
    ("./tmp/snippets/KnuthShuffle.java", "./tmp/snippets/quicksort.py"),
    ("./tmp/snippets/Fibo.java", "./tmp/snippets/knuth_shuffle.py"),
    ("./tmp/snippets/Cipher.java", "./tmp/snippets/fibo.py"),
])
