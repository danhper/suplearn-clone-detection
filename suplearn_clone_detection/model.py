import numpy as np

VOCAB_SIZE = 1000
EMBEDDINGS_DIM = 50
ENCODER_DIM = 30


def create_encoder(input_length,
                   vocab_size=VOCAB_SIZE,
                   word_embeddings_dim=EMBEDDINGS_DIM,
                   encoder_output_dim=ENCODER_DIM):
    from keras import Input
    from keras.layers import Embedding, LSTM

    ast_input = Input(shape=(input_length,), dtype="int32")
    x = Embedding(vocab_size, word_embeddings_dim)(ast_input)
    x = LSTM(encoder_output_dim)(x)
    return (ast_input, x)


def create_model(input_length):
    from keras.models import Model
    from keras.layers import concatenate, Dense

    input_lang1, output_lang1 = create_encoder(input_length)
    input_lang2, output_lang2 = create_encoder(input_length)
    x = concatenate([output_lang1, output_lang2])
    x = Dense(64, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    main_output = Dense(1, activation="sigmoid", name="main_output")(x)
    return Model(inputs=[input_lang1, input_lang2], outputs=main_output)

def try_model():
    from keras import backend as K

    model = create_model(input_length=4)

    actual_input1 = K.variable(np.array([[1, 2, 8, 12], [5, 7, 9, 2]]))
    actual_input2 = K.variable(np.array([[7, 20, 0, 3], [15, 6, 21, 3]]))
    K.eval(model([actual_input1, actual_input2]))
