import os

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from preprocessing import text_preprocessing


def get_test_data(file):
    test_dataset = []
    test_input_data = []
    test_output_data = []
    for folder in os.listdir(file):
        if (folder == 'pos'):
            for data in os.listdir(file + '/' + folder + "/"):
                f = open(file + '/' + folder + '/' + data, 'r')
                data = f.read()
                test_dataset.append({
                    folder: data
                })
                test_input_data.append(data)
                test_output_data.append(folder)
        if (folder == 'neg'):
            for data in os.listdir(file + '/' + folder + "/"):
                f = open(file + '/' + folder + '/' + data, 'r')
                data = f.read()
                test_dataset.append({
                    folder: data
                })
                test_input_data.append(data)
                test_output_data.append(folder)
    return test_input_data, test_output_data


def evaluating_model(Y_true, Y_pred):
    matrix = confusion_matrix(Y_true, Y_pred)
    precision = precision_score(Y_true, Y_pred)
    recall = recall_score(Y_true, Y_pred)
    f1 = f1_score(Y_true, Y_pred)
    return matrix, precision, recall, f1


def test_model(file, model_svm, vect, le):
    test_input_data, test_output_data = get_test_data(file)
    test_input_data = text_preprocessing(test_input_data)
    test_input_data = vect.transform(test_input_data)
    test_output_data = le.transform(test_output_data)

    accuracy = model_svm.score(test_input_data, test_output_data)
    predicted_value = model_svm.predict(test_input_data)
    matrix, precision, recall, f1 = evaluating_model(test_output_data, predicted_value)
    return accuracy, matrix, precision, recall,f1
