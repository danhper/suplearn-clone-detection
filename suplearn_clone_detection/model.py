from typing import Tuple, Optional
import numpy as np

from keras import optimizers
from keras.models import Model, Input
from keras.engine.topology import Layer
from keras.layers import LSTM, Bidirectional, Embedding, concatenate, Dense, multiply


from suplearn_clone_detection import ast_transformer
from suplearn_clone_detection.config import LanguageConfig, ModelConfig
from suplearn_clone_detection.layers import SplitInput, abs_diff, DenseMulti, \
    euclidean_similarity, cosine_similarity, ModelWrapper


def make_embeddings(lang_config: LanguageConfig, index: int):
    embedding_input_size = lang_config.vocabulary_size + lang_config.vocabulary_offset
    kwargs = dict(
        name="embedding_{0}_{1}".format(lang_config.name, index),
        mask_zero=True
    )
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
    ast_input = Input(shape=(None,),
                      dtype="int32", name="input_{0}_{1}".format(lang_config.name, index))

    x = make_embeddings(lang_config, index)(ast_input)

    for i, n in enumerate(lang_config.output_dimensions[:-1]):
        x = create_lstm(n, lang_config, transformer,
                        index=index, position=i + 1, return_sequences=True)(x)

    x = create_lstm(lang_config.output_dimensions[-1], lang_config, transformer,
                    index=index,
                    position=len(lang_config.output_dimensions),
                    return_sequences=False)(x)

    output_dimension = lang_config.output_dimensions[-1]
    if lang_config.bidirectional_encoding:
        output_dimension *= 2

    for i, dim in enumerate(lang_config.hash_dims):
        is_last = i == len(lang_config.hash_dims) - 1
        activation = None if is_last else "relu"
        x = Dense(dim, use_bias=not is_last, activation=activation,
                  name="dense_{0}_{1}_{2}".format(lang_config.name, index, i))(x)
        if is_last:
            output_dimension = dim

    encoder = Model(inputs=ast_input, outputs=x,
                    name="encoder_{0}_{1}".format(lang_config.name, index))
    encoder.output_dimension = output_dimension
    return ast_input, encoder

def make_merge_model(model_config: ModelConfig, input_lang1, input_lang2):
    if model_config.merge_mode == "simple":
        x = concatenate([input_lang1, input_lang2])
    elif model_config.merge_mode == "bidistance":
        hx = multiply([input_lang1, input_lang2])
        hp = abs_diff([input_lang1, input_lang2])
        x = DenseMulti(model_config.merge_output_dim)([hx, hp])
    elif model_config.merge_mode == "euclidean_similarity":
        x = euclidean_similarity([input_lang1, input_lang2],
                                 max_value=model_config.normalization_value)
    elif model_config.merge_mode == "cosine_similarity":
        x = cosine_similarity([input_lang1, input_lang2], min_value=0)
    else:
        raise ValueError("invalid merge mode")

    if model_config.use_output_nn:
        for i, layer_size in enumerate(model_config.dense_layers):
            name = "distance_dense_{0}".format(i)
            x = Dense(layer_size, activation="relu", name=name)(x)
        x = Dense(1, activation="sigmoid", name="main_output")(x)

    model = Model(inputs=[input_lang1, input_lang2], outputs=x,
                  name="merge_model")

    return model


def create_merge_input(lang_config: LanguageConfig,
                       input_dimension: Tuple[Optional[int]],
                       index: int):
    return Input(shape=(input_dimension,),
                 name="encoded_{0}_{1}".format(lang_config.name, index))


def create_model(model_config: ModelConfig):
    lang1_config, lang2_config = model_config.languages

    input_lang1, encoder_lang1 = create_encoder(lang1_config, 1)
    input_lang2, encoder_lang2 = create_encoder(lang2_config, 2)

    lang1_merge_input = create_merge_input(lang1_config, encoder_lang1.output_dimension, 1)
    lang2_merge_input = create_merge_input(lang2_config, encoder_lang2.output_dimension, 2)
    merge_model = make_merge_model(model_config, lang1_merge_input, lang2_merge_input)

    output_lang1 = encoder_lang1(input_lang1)
    output_lang2 = encoder_lang2(input_lang2)

    result = merge_model([output_lang1, output_lang2])

    model = ModelWrapper(inputs=[input_lang1, input_lang2], outputs=result)
    optimizer_class = getattr(optimizers, model_config.optimizer["type"])
    optimizer = optimizer_class(**model_config.optimizer.get("options", {}))
    model.compile(optimizer=optimizer, loss=model_config.loss, metrics=model_config.metrics)

    return model
