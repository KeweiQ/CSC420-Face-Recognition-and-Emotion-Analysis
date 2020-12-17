'''
Module for evaluating the performance of several models
'''


import keras
import numpy as np

from keras.utils import to_categorical
from sklearn.metrics import classification_report


def evaluateModel(model_trained, model_kind, test, test_label, algorithm):
    """
    The function for evaluating the input model with the input test dataset and
    corresponding labels for the test dataset.

    Args:
        model_trained:     the input trained model for the purpose of evaluating.
        model_kind:        kind of the input model.
        test:              the test dataset.
        test_label:        the labels for the test dataset.
        algorithm:         the feature algorithm that we want to perform.

    Returns:
        None.

    Notes: we will show the performance of each model in several ways.
    """
    if model_kind == 'cnn':
        if algorithm == 'eigenfaces':
            # Reshape the input datasets to accord the cnn model
            test = test.reshape(-1, 1, 625)

            # One-hot encoding for the test labels
            test_label_hotcoder = to_categorical(test_label)

            # Use the model to predict
            test_pred = model_trained.predict(test)
            # Print classification report
            print(classification_report(np.argmax(test_label_hotcoder, axis=1), np.argmax(test_pred, axis=1)))

        elif algorithm == 'fisherfaces':

            # Reshape the input datasets to accord the cnn model
            test = test.reshape(-1, 1, 6)

            # One-hot encoding for the test labels
            test_label_hotcoder = to_categorical(test_label)

            # Use the model to predict
            test_pred = model_trained.predict(test)
            # Print classification report
            print(classification_report(np.argmax(test_label_hotcoder, axis=1), np.argmax(test_pred, axis=1)))

    elif model_kind == 'svm':
        pass
    elif model_kind == 'adaboost':
        pass
