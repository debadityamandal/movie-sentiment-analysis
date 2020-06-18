# -*- coding: utf-8 -*-
import os

from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from preprocessing import text_preprocessing


def create_input_output(file):
    input_data = []
    output_data = []
    for folder in os.listdir(file):
        if (folder == 'pos'):
            for data in os.listdir(file + '/' + folder + "/"):
                f = open(file + '/' + folder + '/' + data, 'r')
                data = f.read()
                input_data.append(data)
                output_data.append(folder)
        if (folder == 'neg'):
            for data in os.listdir(file + '/' + folder + "/"):
                f = open(file + '/' + folder + '/' + data, 'r')
                data = f.read()
                input_data.append(data)
                output_data.append(folder)
    return input_data, output_data


def vectorize(input_data):
    vect = TfidfVectorizer()
    vect = vect.fit(input_data)
    input_data = vect.transform(input_data)
    return vect, input_data


def label_encoding(output_data):
    le = LabelEncoder()
    output_data = le.fit_transform(output_data)
    return le, output_data


def model_support_vector_machine(input_data, output_data):
    model_svm = svm.SVC(gamma='scale')
    model_svm.fit(input_data, output_data)
    return model_svm


def train_model(file):
    input_data, output_data = create_input_output(file)
    input_data = text_preprocessing(input_data)
    vect, input_data = vectorize(input_data)
    le, output_data = label_encoding(output_data)

    print("Training SVM model:\n")
    model_svm = model_support_vector_machine(input_data, output_data)
    print("SVM model trained successfully\n")
    accuracy = model_svm.score(input_data, output_data)
    return accuracy, model_svm, vect, le
