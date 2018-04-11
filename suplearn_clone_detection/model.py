from os import path

import numpy as np

from keras import optimizers
from keras.models import Model, Input, Layer
from keras.layers import LSTM, Bidirectional, Embedding, concatenate, Dense, multiply


from suplearn_clone_detection import ast_transformer
from suplearn_clone_detection.config import LanguageConfig, ModelConfig
from suplearn_clone_detection.layers import SplitInput, abs_diff, DenseMulti, \
    euclidean_similarity, cosine_similarity, ModelWrapper



def make_embeddings(lang_config: LanguageConfig, index: int):
    embedding_input_size = lang_config.vocabulary_size + lang_config.vocabulary_offset
    kwargs = {"name": "embedding_{0}_{1}".format(lang_config.name, index)}
    if lang_config.embeddings:
        weights = np.load(lang_config.embeddings)
        padding = np.zeros((lang_config.vocabulary_offset, lang_config.embeddings_dimension))
        kwargs["weights"] = [np.vstack([padding, weights])]

    return Embedding(embedding_input_size, lang_config.embeddings_dimension, **kwargs)


def create_lstm(output_dimension: int,
                lang_config: LanguageConfig,
                transformer: ast_transformer.ASTTransformer,
                index: int,
                position: int,
                return_sequences: bool) -> Layer:

    lstm = LSTM(output_dimension,
                return_sequences=return_sequences,
                name="lstm_{0}_{1}_{2}".format(lang_config.name, index, position))

    if transformer.split_input:
        lstm = SplitInput(
            lstm, name="bidfs_lstm_{0}_{1}_{2}".format(lang_config.name, index, position))

    if lang_config.bidirectional_encoding:
        if transformer.split_input:
            raise ValueError("bidirectional_encoding cannot be used with {0}".format(
                lang_config.transformer_class_name))
        lstm = Bidirectional(
            lstm, name="bilstm_{0}_{1}_{2}".format(lang_config.name, index, position))

    return lstm


def create_encoder(lang_config: LanguageConfig, index: int):
    transformer = ast_transformer.create(lang_config)

    ast_input = Input(shape=(transformer.total_input_length,),
                      dtype="int32", name="input_{0}_{1}".format(lang_config.name, index))

    x = make_embeddings(lang_config, index)(ast_input)

    for i, n in enumerate(lang_config.output_dimensions[:-1]):
        x = create_lstm(n, lang_config, transformer,
                        index=index, position=i + 1, return_sequences=True)(x)

    x = create_lstm(lang_config.output_dimensions[-1], lang_config, transformer,
                    index=index,
                    position=len(lang_config.output_dimensions),
                    return_sequences=False)(x)

    if lang_config.hash_dim:
        x = Dense(lang_config.hash_dim, use_bias=False)(x)

    encoder = Model(inputs=ast_input, outputs=x, name="encoder_{0}".format(index))
    return ast_input, encoder


def create_model(model_config: ModelConfig):
    lang1_config, lang2_config = model_config.languages
    input_lang1, encoder_lang1 = create_encoder(lang1_config, 1)
    input_lang2, encoder_lang2 = create_encoder(lang2_config, 2)
    output_lang1 = encoder_lang1(input_lang1)
    output_lang2 = encoder_lang2(input_lang2)

    if model_config.merge_mode == "simple":
        x = concatenate([output_lang1, output_lang2])
    elif model_config.merge_mode == "bidistance":
        hx = multiply([output_lang1, output_lang2])
        hp = abs_diff([output_lang1, output_lang2])
        x = DenseMulti(model_config.merge_output_dim)([hx, hp])
    elif model_config.merge_mode == "euclidean_similarity":
        x = euclidean_similarity([output_lang1, output_lang2],
                                 max_value=model_config.normalization_value)
    elif model_config.merge_mode == "cosine_similarity":
        x = cosine_similarity([output_lang1, output_lang2], min_value=0)
    else:
        raise ValueError("invalid merge mode")

    if model_config.indexable_output:
        main_output = x
    else:
        for layer_size in model_config.dense_layers:
            x = Dense(layer_size, activation="relu")(x)
            main_output = Dense(1, activation="sigmoid", name="main_output")(x)

    model = ModelWrapper(inputs=[input_lang1, input_lang2], outputs=main_output)
    optimizer_class = getattr(optimizers, model_config.optimizer["type"])
    optimizer = optimizer_class(**model_config.optimizer.get("options", {}))
    model.compile(optimizer=optimizer, loss=model_config.loss, metrics=model_config.metrics)

    return model
