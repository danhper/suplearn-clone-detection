from os import path

import numpy as np

import torch
from torch import nn, optim
from torch.autograd import Variable

from suplearn_clone_detection.ast_loader import ASTLoader
from suplearn_clone_detection.config import Config
from suplearn_clone_detection import ast_transformer
from suplearn_clone_detection.pytorch_model import Encoder, CloneDetector
from suplearn_clone_detection.data_generator import DataGenerator

java_ast_paths = path.expanduser("~/workspaces/research/dataset/atcoder/asts/java-asts-small.jsonl")
java_input_files = ["src/b/4/0/1381956.java", "src/b/22/0/1133780.java"]

python_ast_paths = path.expanduser("~/workspaces/research/dataset/atcoder/asts/python-asts-small.jsonl")
python_input_files = ["src/r/81/4/1526561.py", "src/r/75/2/1323648.py"]

config = Config.from_file("config.yml")
transformers = ast_transformer.create_all(config.model.languages)

def get_input(filepath, input_files, index):
    trans = transformers[index]
    loader = ASTLoader(filepath)
    sample_asts = [trans.transform_ast(loader.get_ast(v)) for v in input_files]
    return Variable(torch.from_numpy(np.array(sample_asts, dtype=np.long)))

java_encoder = Encoder(config.model.languages[0])
java_input = get_input(java_ast_paths, java_input_files, 0)

java_output = java_encoder(java_input)
print(java_output.data.shape)

python_encoder = Encoder(config.model.languages[1])
python_input = get_input(python_ast_paths, python_input_files, 1)

python_output = python_encoder(python_input)
print(python_output.data.shape)


detector = CloneDetector(config.model)
result = detector(java_input, python_input)
print(result)

data_generator = DataGenerator(
    config.generator, ast_transformer.create_all(config.model.languages))
it = data_generator.make_iterator()


loss_fn = nn.BCELoss()
optimizer = optim.RMSprop(detector.parameters())
inputs, targets, weights = it.next_batch(64)
inputs = [Variable(torch.from_numpy(i)) for i in inputs]
targets = Variable(torch.from_numpy(targets).type(torch.FloatTensor))
output = detector(*inputs)
loss_fn(output, targets).backward()
optimizer.step()


result = detector(java_input, python_input)
print(result)