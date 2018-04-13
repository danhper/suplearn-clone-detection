import torch
from torch import nn

import numpy as np

from suplearn_clone_detection.config import LanguageConfig


class Encoder(nn.Module):
    def __init__(self, lang_config: LanguageConfig):
        super(Encoder, self).__init__()
        self.embeddings = nn.Embedding(
            lang_config.vocabulary_size + lang_config.vocabulary_offset,
            lang_config.embeddings_dimension,
            padding_idx=0 if lang_config.vocabulary_offset == 1 else None
        )
        if lang_config.embeddings:
            pretrained = np.load(lang_config.embeddings)
            padding = np.zeros((lang_config.vocabulary_offset, lang_config.embeddings_dimension))
            weights = torch.Tensor(np.vstack([padding, pretrained]))
            self.embeddings.weight.data.copy_(weights)

    def forward(self, x):
        x = self.embeddings(x)

        return x
