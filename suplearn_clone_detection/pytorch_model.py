import torch
from torch import nn

import numpy as np

from suplearn_clone_detection.config import LanguageConfig


class Encoder(nn.Module):
    def __init__(self, config: LanguageConfig):
        super(Encoder, self).__init__()
        self.config = config
        self._build_embedding()
        self.lstm = nn.LSTM(
            input_size=self.config.embeddings_dimension,
            hidden_size=config.output_dimensions[0],
            num_layers=len(config.output_dimensions),
            bidirectional=config.bidirectional_encoding,
            batch_first=True,
        )
        self.hidden = None

    def reset(self):
        self.zero_grad()
        self.hidden = None

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

        return x
