import pickle

import tensorflow as tf
from sklearn.model_selection import train_test_split

from load_dataset import read_file
from test import evaluation
from train import train_sequence_model

# Training model
train_texts, train_labels = read_file('train.ft.txt.bz2')
X_train, X_val, Y_train, Y_val = train_test_split(train_texts, train_labels, test_size=0.1, shuffle=True,
                                                  random_state=42)
history = train_sequence_model(X_train, Y_train, X_val, Y_val)
print("Training finished\n")

# Evaluate model
model = tf.keras.models.load_model('model_data/amazon_review_sentiment_model.h5')
with open('model_data/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
test_texts, test_labels = read_file('test.ft.txt.bz2')
accuracy, confusion_matrix, precision, recall, f1_score = evaluation(test_texts, test_labels, model, tokenizer)
