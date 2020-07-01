from tensorflow.python.keras import models
from tensorflow.python.keras.layers import Dense, Dropout, Embedding, Bidirectional, LSTM


def _get_last_layer_units_and_activation(num_classes):
    if num_classes == 2:
        activation = 'sigmoid'
        units = 1
    else:
        activation = 'softmax'
        units = num_classes
    return units, activation


def bidirectional_lstm_model(embedding_dim, dropout_rate, max_len, num_classes, num_features):
    op_units, op_activation = _get_last_layer_units_and_activation(num_classes)
    model = models.Sequential()
    model.add(Embedding(input_dim=num_features, output_dim=embedding_dim, input_length=max_len))
    model.add(Bidirectional(LSTM(128)))
    #     model.add(LSTM(128,return_sequences=True))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(op_units, activation=op_activation))
    return model
