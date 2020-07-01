import pickle

from test import test_model
from train import train_model

training_accuracy, model, vect, le = train_model("/home/debaditya/Desktop/movie-sentiment-analysis/imdb_movie_review/aclImdb/train")
print("Training Accuracy:", training_accuracy)
with open('/home/debaditya/Desktop/movie-sentiment-analysis/model_data/model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('/home/debaditya/Desktop/movie-sentiment-analysis/model_data/vect.pkl', 'wb') as f:
    pickle.dump(vect, f)
with open('/home/debaditya/Desktop/movie-sentiment-analysis/model_data/le.pkl', 'wb') as f:
    pickle.dump(le, f)
print("Model has stored\n")
testing_accuracy, confusion_matrix, precision, recall, f1 = test_model("/home/debaditya/Desktop/movie-sentiment-analysis/imdb_movie_review/aclImdb/test", model, vect,
                                                                       le)
print("Testing Accuracy:", testing_accuracy)
print("Confusion Matrix:", confusion_matrix)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
