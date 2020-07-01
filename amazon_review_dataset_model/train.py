import pickle

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.preprocessing import text, sequence

from model import bidirectional_lstm_model

max_sequence_length = 500
maximum_features = 20000
FLAGS = None


def sequence_vectorizer(X_train, X_val):
    tokenizer = text.Tokenizer(num_words=maximum_features)
    tokenizer.fit_on_texts(X_train)
    X_train = tokenizer.texts_to_sequences(X_train)
    X_val = tokenizer.texts_to_sequences(X_val)
    maximum_length = len(max(X_train, key=len))
    if (maximum_length > max_sequence_length):
        maximum_length = max_sequence_length
    print(maximum_length)
    X_train = sequence.pad_sequences(X_train, maxlen=maximum_length)
    X_val = sequence.pad_sequences(X_val, maxlen=maximum_length)
    return X_train, X_val, tokenizer.word_index, tokenizer


def change_shape(Y_train, Y_val):
    Y_train = np.array(Y_train)
    Y_val = np.array(Y_val)
    Y_train = Y_train.reshape((Y_train.shape[0], 1))
    Y_val = Y_val.reshape((Y_val.shape[0], 1))
    return Y_train, Y_val


def train_sequence_model(X_train, Y_train, X_val, Y_val, learning_rate=1e-3, epochs=100, batch_size=1024,
                         dropout_rate=0.2, embedding_dim=200):
    Y_train, Y_val = change_shape(Y_train, Y_val)
    X_train, X_val, word_index, tokenizer = sequence_vectorizer(X_train, X_val)
    print("Vectorization completed")
    num_features = min(len(word_index) + 1, maximum_features)
    model = bidirectional_lstm_model(embedding_dim, dropout_rate, X_train.shape[1], 2, num_features)
    loss = 'binary_crossentropy'
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)]
    history = model.fit(X_train, Y_train, epochs=epochs, validation_data=(X_val, Y_val), callbacks=callbacks, verbose=2,
                        batch_size=batch_size)
    model.save('model_data/amazon_review_sentiment_model.h5')
    with open('model_data/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    return history.history
