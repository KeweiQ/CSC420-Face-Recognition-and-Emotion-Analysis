
"""
This file includes:
- The implementation of the neural network model used to train the model;
- The prediction of test model and result analysis.
"""


from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import preprocess_dataset


if __name__ == '__main__':
    # Load and split the dataset
    dataset_list = preprocess_dataset.load_dataset('./CK+48')
    img_train, img_train_label, img_validation, img_validation_label, img_test, img_test_label = \
        preprocess_dataset.split_data(dataset_list)

