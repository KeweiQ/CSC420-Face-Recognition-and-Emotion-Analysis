"""
This file includes:
- The implementation of the neural network model used to train the model;
- The prediction of test model and result analysis.
"""


# from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import numpy as np
from feature_extraction_Eigenfaces_Fisherfaces.py import fisherfaces
from matplotlib import pyplot as plt

import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

import preprocess_dataset


if __name__ == '__main__':
    # Load and split the dataset
    dataset_list = preprocess_dataset.load_dataset('./CK+48')
    img_train, img_train_label, img_validation, img_validation_label, img_test, img_test_label, le = \
        preprocess_dataset.split_data(dataset_list)

    # Perform face detection dimensionality reduction on the datasets
    # TODO: Implement this
    fisher_train, fisher_test, fisher_validation = fe.fisherfaces(img_train, img_test, img_validation, img_train_label)
    img_train_reduced = fisher_train
    img_validation_reduced = fisher_validation
    img_test_reduced = fisher_test

    # Construct the convolutional neural network
    cnn_clf = Sequential()
    # The first layer
    cnn_clf.add(Conv2D(32, kernel_size=(3, 3), activation='linear', input_shape=(28, 28, 1), padding='same'))
    cnn_clf.add(LeakyReLU(alpha=0.1))
    cnn_clf.add(MaxPooling2D((3, 3), padding='same'))
    cnn_clf.add(Dropout(0.25))
    # The second layer
    cnn_clf.add(Conv2D(64, kernel_size=(3, 3), activation='linear', padding='same'))
    cnn_clf.add(LeakyReLU(alpha=0.1))
    cnn_clf.add(MaxPooling2D((3, 3), padding='same'))
    cnn_clf.add(Dropout(0.25))
    # The third layer
    cnn_clf.add(Conv2D(128, kernel_size=(3, 3), activation='linear', padding='same'))
    cnn_clf.add(LeakyReLU(alpha=0.1))
    cnn_clf.add(MaxPooling2D((3, 3), padding='same'))
    cnn_clf.add(Dropout(0.4))
    # The first dense layer
    cnn_clf.add(Flatten())
    cnn_clf.add(Dense(128, activation='linear'))
    # The second dense layer to complete categorization
    cnn_clf.add(LeakyReLU(alpha=0.1))
    cnn_clf.add(Dropout(0.3))
    cnn_clf.add(Dense(7, activation='softmax'))

    # Compile the neural network model
    cnn_clf.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.Adam(),
                    metrics=['accuracy']
                    )

    # Train the neural network model
    cnn_train = cnn_clf.fit(img_train_reduced, img_train_label,
                            batch_size=256,
                            epochs=20,
                            verbose=1,
                            validation_data=(img_validation_reduced, img_validation_label)
                            )

    # Test the model
    test_pred = cnn_clf.predict(img_test_reduced)
    # Print classification report
    print(classification_report(img_test_label, test_pred))

    # Plot a gallery of 20 sample results
    for i in range(20):
        plt.imshow(img_test[i], cmap=plt.cm.gray)
        plt.title("Predicted: {}\nTrue: {}".format(test_pred[i], img_test_label[i]))
        plt.show()