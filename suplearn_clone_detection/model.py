import numpy as np

from suplearn_clone_detection.config import Config


def create_encoder(lang_config):
    from keras import Input
    from keras.layers import Embedding, LSTM, Bidirectional

    embedding_input_size = lang_config.vocabulary_size + lang_config.vocabulary_offset

    ast_input = Input(shape=(lang_config.input_length,), dtype="int32")
    x = Embedding(embedding_input_size, lang_config.embeddings_dimension)(ast_input)
    lstm = LSTM(lang_config.output_dimension)
    if lang_config.bidirectional_encoding:
        lstm = Bidirectional(lstm)
    x = lstm(x)
    return (ast_input, x)


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


def try_model():
    from keras import backend as K

    config = Config.from_file("config.yml")
    for lang in config.model.languages:
        lang.vocabulary_size = 100

    model = create_model(config.model)

    left_input = np.array([[1, 2, 8, 12], [5, 7, 9, 2]])
    right_input = np.array([[7, 20, 0, 3], [15, 6, 21, 3]])
    model.train_on_batch([left_input, right_input], np.array([[0], [1]]))
    K.eval(model([K.variable(left_input), K.variable(right_input)]))
