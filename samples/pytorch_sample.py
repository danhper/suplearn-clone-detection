from os import path

import numpy as np

import torch
from torch.autograd import Variable

from suplearn_clone_detection.ast_loader import ASTLoader
from suplearn_clone_detection.config import Config
from suplearn_clone_detection import ast_transformer
from suplearn_clone_detection.pytorch_model import Encoder

ast_paths = path.expanduser("~/workspaces/research/dataset/atcoder/asts/asts-small.jsonl")
input_files = ["src/b/4/0/1381956.java", "src/b/22/0/1133780.java"]


config = Config.from_file("config.yml")
lang_config = config.model.languages[0]

transformers = ast_transformer.create_all(config.model.languages)
trans = transformers[0]

loader = ASTLoader(ast_paths)
sample_asts = [trans.transform_ast(loader.get_ast(v)) for v in input_files]

encoder = Encoder(lang_config)

model_input = Variable(torch.from_numpy(np.array(sample_asts, dtype=np.long)))
output = encoder(model_input)
print(output.data.shape)
