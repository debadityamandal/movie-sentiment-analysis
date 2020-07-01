import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from tensorflow.python.keras.preprocessing import sequence


def vectorize(test_texts, tokenizer,max_len=255):
    test_texts = tokenizer.texts_to_sequences(test_texts)
    test_texts = sequence.pad_sequences(test_texts, maxlen=max_len)
    return test_texts


def change_shape(test_labels):
    test_labels = np.array(test_labels)
    test_labels = test_labels.reshape((test_labels.shape[0], 1))
    return test_labels


def evaluation(test_texts, test_labels, model, tokenizer):
    test_texts = vectorize(test_texts, tokenizer)
    test_labels = change_shape(test_labels)
    test_accuracy = model.evaluate(test_texts, test_labels)
    predicted_classes = model.predict_classes(test_texts)
    return test_accuracy, confusion_matrix(test_labels, predicted_classes), precision_score(test_labels,
                                                                                            predicted_classes), recall_score(
        test_labels, predicted_classes), f1_score(test_labels, predicted_classes)
