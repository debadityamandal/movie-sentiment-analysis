# movie-sentiment-analysis
This code will analyze sentiment of the viewers by their review about the movie.This model supports only English language.
# Dataset
I downloaded movie review dataset from following link-
http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
# Model Evaluation
    Training accuracy:- ~99%
    Testing accuracy:- ~88.636%
    Precision:- ~89%
    Recall:- ~88%
    F1 Score:- ~89%
# File Description
    collect_dataset.py will help to collect dataset
    train.py, test.py, train_sentiment_model.py will train the model with downloaded dataset
    live_demo.py will help to test the model with new review. User can give any review here.
    model_data folder is containing pre-trained model
# Execution Order
    If you want to use our pre-trained model then download pre-trained model from model_data folder and execute live_demo.py.
    If you want to train your custom model then please maintain following order-
    1. Execute collect_dataset.py
    2. Execute train_sentiment_model.py
    3. Execute live_demo.py
    N.B:- This is CPU version.GPU is not required for now.