import numpy as np


def make_embeddings(lang_config):
    from keras.layers import Embedding

    embedding_input_size = lang_config.vocabulary_size + lang_config.vocabulary_offset
    kwargs = {"name": "embedding_{0}".format(lang_config.name)}
    if lang_config.embeddings:
        weights = np.load(lang_config.embeddings)
        padding = np.zeros((lang_config.vocabulary_offset, lang_config.embeddings_dimension))
        kwargs["weights"] = [np.vstack([padding, weights])]

    return Embedding(embedding_input_size, lang_config.embeddings_dimension, **kwargs)


def create_encoder(lang_config):
    from keras import Input
    from keras.layers import LSTM, Bidirectional

    ast_input = Input(shape=(lang_config.input_length,),
                      dtype="int32", name="input_{0}".format(lang_config.name))
    x = make_embeddings(lang_config)(ast_input)
    lstm = LSTM(lang_config.output_dimension, name="lstm_{0}".format(lang_config.name))
    if lang_config.bidirectional_encoding:
        lstm = Bidirectional(lstm, name="bilstm_{0}".format(lang_config.name))
    x = lstm(x)
    return ast_input, x


def create_model(model_config):
    from keras.models import Model
    from keras.layers import concatenate, Dense
    from keras import optimizers

    lang1_config, lang2_config = model_config.languages
    input_lang1, output_lang1 = create_encoder(lang1_config)
    input_lang2, output_lang2 = create_encoder(lang2_config)

    x = concatenate([output_lang1, output_lang2])
    for layer_size in model_config.dense_layers:
        x = Dense(layer_size, activation="relu")(x)
    main_output = Dense(1, activation="sigmoid", name="main_output")(x)

    model = Model(inputs=[input_lang1, input_lang2], outputs=main_output)
    optimizer_class = getattr(optimizers, model_config.optimizer["type"])
    optimizer = optimizer_class(**model_config.optimizer.get("options", {}))
    model.compile(optimizer=optimizer,
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model
