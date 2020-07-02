# Introduction
This repository is containing Two different models trained by two different datasets and different NLP techniques
    - stanford_dataset_model is trained with stanford dataset
    - amazon_review_dataset_model is trained with amazon review dataset
## stanford_dataset_model
This directory is containing codes which will analyze sentiment of the viewers by their review about the movie.This model supports only English language.
# Dataset
I downloaded movie review dataset from http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
# Model Evaluation
    Training accuracy:- ~99%
    Testing accuracy:- ~88.636%
    Precision:- ~89%
    Recall:- ~88%
    F1 Score:- ~89%
# File Description
    collect_dataset.py will help to collect dataset
    train.py, test.py, train_sentiment_model.py will train and evaluate the model with downloaded dataset
    live_demo.py will help to test the model with new review. User can give any review here.
    model_data folder is containing pre-trained model
# Execution Order
    If you want to use our pre-trained model then download all files from model_data folder and execute live_demo.py.
    If you want to train your custom model then please maintain following order-
    1. Execute collect_dataset.py
    2. Execute train_sentiment_model.py
    3. Execute live_demo.py

## amazon_review_dataset_model
This directory is containing codes which will analyze sentiment of any type of review including movie reviews. This model supports only English language.
# Dataset
I used amazon review dataset and I downloaded it from https://www.kaggle.com/bittlingmayer/amazonreviews
# Model Evaluation
    Training Accuracy:- 0.9752
    Validation Accuracy:- 0.9521
    Testing Accuracy:- 0.9514
    Precision:- 0.9496
    Recall:- 0.9534
    F1-score:- 0.9515
# File Description
    load_dataset.py will help to read dataset
    model.py is describing the model
    train.py is training model with downloaded data
    test.py is evaluating the trained model with unknown labeled data
    train_sentiment_model.py is accessing above mentioned python files to create and evaluate model
    live_demo.py will give you predicted class of any given review
# Execution order
    If you want to use our pre-trained model then download all files from model_data folder and execute live_demo.py
    If you to train your model then follow below steps-
    1. Download dataset by clicking on the above mentioned link
    2. Execute train_sentiment_model.py
    3. Execute live_demo.py to access model with custom review
If you find any error/bug or if you think I need to modify the code to make it fast or to increase performance of the model then please raise issue.
