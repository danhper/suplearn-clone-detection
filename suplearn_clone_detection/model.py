import numpy as np

VOCAB_SIZE = 1000
EMBEDDINGS_DIM = 50
ENCODER_DIM = 30


class EncoderOptions:
    def __init__(self, options):
        self.input_length = options["input_length"]
        self.vocabulary_size = options["vocabulary_size"]
        self.embeddings_dimension = options["embeddings_dimension"]
        self.output_dimension = options["output_dimension"]


class ModelOptions:
    def __init__(self, left_encoder, right_encoder):
        self.left_encoder = left_encoder
        self.right_encoder = right_encoder


def create_encoder(encoder_options):
    from keras import Input
    from keras.layers import Embedding, LSTM

    ast_input = Input(shape=(encoder_options.input_length,), dtype="int32")
    x = Embedding(encoder_options.vocabulary_size,
                  encoder_options.embeddings_dimension)(ast_input)
    x = LSTM(encoder_options.output_dimension)(x)
    return (ast_input, x)


def create_model(model_options):
    from keras.models import Model
    from keras.layers import concatenate, Dense

    input_lang1, output_lang1 = create_encoder(model_options.left_encoder)
    input_lang2, output_lang2 = create_encoder(model_options.right_encoder)
    x = concatenate([output_lang1, output_lang2])
    x = Dense(64, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    main_output = Dense(1, activation="sigmoid", name="main_output")(x)
    return Model(inputs=[input_lang1, input_lang2], outputs=main_output)

def try_model():
    from keras import backend as K

    left_encoder_options = EncoderOptions(dict(
        input_length=4, vocabulary_size=VOCAB_SIZE,
        embeddings_dimension=EMBEDDINGS_DIM, output_dimension=ENCODER_DIM))
    right_encoder_options = EncoderOptions(dict(
        input_length=4, vocabulary_size=VOCAB_SIZE,
        embeddings_dimension=EMBEDDINGS_DIM, output_dimension=ENCODER_DIM))
    model_options = ModelOptions(left_encoder_options, right_encoder_options)
    model = create_model(model_options)
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

    left_input = K.variable(np.array([[1, 2, 8, 12], [5, 7, 9, 2]]))
    right_input = np.array([[7, 20, 0, 3], [15, 6, 21, 3]])
    model.train_on_batch([np.array([[7, 20, 0, 3], [15, 6, 21, 3]]), np.array([[7, 20, 0, 3], [15, 6, 21, 3]])], np.array([0, 1]))
    K.eval(model([left_input, right_input]))
