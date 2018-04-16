import math

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

from suplearn_clone_detection.config import LanguageConfig, ModelConfig


class Encoder(nn.Module):
    def __init__(self, config: LanguageConfig):
        super(Encoder, self).__init__()
        self.config = config
        self.hidden = None
        self.encoder_output_dimension = self.config.output_dimensions[0]
        if self.config.bidirectional_encoding:
            self.encoder_output_dimension *= 2
        if self.config.hash_dims:
            self.output_dimension = self.config.hash_dims[-1]
        else:
            self.output_dimension = self.encoder_output_dimension

        self._build_embedding()
        self._build_hash_layers()
        self.lstm = nn.LSTM(
            input_size=self.config.embeddings_dimension,
            hidden_size=self.config.output_dimensions[0],
            num_layers=len(self.config.output_dimensions),
            bidirectional=self.config.bidirectional_encoding,
            batch_first=True,
        )

    def reset(self):
        self.zero_grad()
        self.hidden = None

    def _build_hash_layers(self):
        input_size = self.encoder_output_dimension
        self.hash_layers = []
        for i, dim in enumerate(self.config.hash_dims):
            bias = i != len(self.config.hash_dims) - 1
            self.hash_layers.append(nn.Linear(input_size, dim, bias=bias))
            input_size = dim

    def _build_embedding(self):
        self.vocab_size = self.config.vocabulary_size
        padding_idx = None
        if self.config.vocabulary_offset == 1:
            padding_idx = 0
            self.vocab_size += 1
        self.embeddings = nn.Embedding(
            self.vocab_size, self.config.embeddings_dimension, padding_idx)

        if self.config.embeddings:
            pretrained = np.load(self.config.embeddings)
            padding = np.zeros((self.config.vocabulary_offset, self.config.embeddings_dimension))
            weights = torch.Tensor(np.vstack([padding, pretrained]))
            self.embeddings.weight.data.copy_(weights)

    def forward(self, x):
        x = self.embeddings(x)
        x, self.hidden = self.lstm(x)
        x = x[:, -1, :]
        for i, layer in enumerate(self.hash_layers):
            x = layer(x)
            if i != len(self.hash_layers) - 1:
                x = F.relu(x)

        return x


class BiDistanceMerge(nn.Module):
    def __init__(self, left_dim, right_dim, output_dim):
        super(BiDistanceMerge, self).__init__()
        self.left_weight = Parameter(torch.Tensor(output_dim, left_dim))
        self.right_weight = Parameter(torch.Tensor(output_dim, right_dim))
        self.bias = Parameter(torch.Tensor(output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.left_weight.size(1))
        self.left_weight.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.right_weight.size(1))
        self.right_weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, left, right):
        result = left.matmul(self.left_weight.t())
        result += right.matmul(self.right_weight.t())
        result += self.bias
        return result


class CloneDetector(nn.Module):
    def __init__(self, config: ModelConfig):
        super(CloneDetector, self).__init__()
        self.config = config
        self.encoder_left = Encoder(config.languages[0])
        self.encoder_right = Encoder(config.languages[1])
        self._build_merge_layer()
        self._build_output_network()

    @staticmethod
    def concat(left, right):
        return torch.cat((left, right), dim=1)

    def _build_merge_layer(self):
        left_dim = self.encoder_left.output_dimension
        right_dim = self.encoder_right.output_dimension
        if self.config.merge_mode == "concat":
            self.merge = self.concat
            self.concat_dimension = left_dim + right_dim
        elif self.config.merge_mode == "bidistance":
            self.merge = BiDistanceMerge(left_dim, right_dim, self.config.merge_output_dim)
            self.concat_dimension = self.config.merge_output_dim
        else:
            raise ValueError("unknown merge mode {0}".format(self.config.merge_mode))

    def _build_output_network(self):
        input_size = self.concat_dimension
        self.dense_layers = []
        for dim in self.config.dense_layers:
            self.dense_layers.append(nn.Linear(input_size, dim))
            input_size = dim

    def forward(self, left_input, right_input):
        left = self.encoder_left(left_input)
        right = self.encoder_left(right_input)
        x = self.merge(left, right)
        for i, layer in enumerate(self.dense_layers):
            x = layer(x)
            if i == len(self.dense_layers) - 1:
                x = F.sigmoid(x)
            else:
                x = F.relu(x)

        return x
