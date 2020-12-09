
"""
This file includes:
- The implementation of the neural network model used to train the model;
- The prediction of test model and result analysis.
"""


from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import numpy as np
import preprocess_dataset


if __name__ == '__main__':
    # Load and split the dataset
    dataset_list = preprocess_dataset.load_dataset('./CK+48')
    img_train, img_train_label, img_validation, img_validation_label, img_test, img_test_label = \
        preprocess_dataset.split_data(dataset_list)

    # Perform dimensionality reduction on the datasets
    # TODO: Implement this
    img_train_reduced = img_train
    img_validation_reduced = img_validation
    img_test_reduced = img_test

    # Train the neural network
    mlp_clf = MLPClassifier(solver='lbfgs', early_stopping=True).fit(img_train_reduced, img_train_label)

    # Predict the validation set to check model accuracy
    validation_pred = mlp_clf.predict(img_validation_reduced)
    validation_accuracy = np.average(np.array(
        [1 if validation_pred[i] == img_validation_label[i] else 0 for i in range(len(validation_pred))]
    ))
