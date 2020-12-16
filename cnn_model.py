"""
This file includes:
- The implementation of the neural network model used to train the model;
- The prediction of test model and result analysis.
"""


# from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import numpy as np
import feature_extraction_fisherfaces_eigenfaces as fe
from matplotlib import pyplot as plt

import keras
from keras.utils import to_categorical
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

import preprocess_dataset


if __name__ == '__main__':
    # Load and split the dataset
    dataset_list = preprocess_dataset.load_dataset('CK+48')
    img_train, img_train_label, img_validation, img_validation_label, img_test, img_test_label, le = \
        preprocess_dataset.split_data(dataset_list)

    # Perform face detection dimensionality reduction on the datasets
    img_train_reduced, img_test_reduced, img_validation_reduced = \
        fe.fisherfaces(img_train, img_test, img_validation, img_train_label, le)

    img_train_reduced = img_train_reduced.reshape(-1, 1, 6)
    img_validation_reduced = img_validation_reduced.reshape(-1, 1, 6)
    img_test_reduced = img_test_reduced.reshape(-1, 1, 6)

    img_train_label = to_categorical(img_train_label)
    img_validation_label = to_categorical(img_validation_label)
    img_test_label = to_categorical(img_test_label)

    # Construct the convolutional neural network
    cnn_clf = Sequential()

    # The first layer
    cnn_clf.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(1, 6), padding='same'))
    cnn_clf.add(LeakyReLU(alpha=0.1))
    cnn_clf.add(MaxPooling1D(3, padding='same'))
    cnn_clf.add(Dropout(0.25))

    # The second layer
    cnn_clf.add(Conv1D(64, kernel_size=3, activation='relu', padding='same'))
    cnn_clf.add(LeakyReLU(alpha=0.1))
    cnn_clf.add(MaxPooling1D(3, padding='same'))
    cnn_clf.add(Dropout(0.25))

    # The third layer
    cnn_clf.add(Conv1D(128, kernel_size=3, activation='relu', padding='same'))
    cnn_clf.add(LeakyReLU(alpha=0.1))
    cnn_clf.add(MaxPooling1D(3, padding='same'))
    cnn_clf.add(Dropout(0.4))

    # The first dense layer
    cnn_clf.add(Flatten())
    cnn_clf.add(Dense(128, activation='relu'))
    
    # The second dense layer to complete categorization
    cnn_clf.add(LeakyReLU(alpha=0.1))
    cnn_clf.add(Dropout(0.3))
    cnn_clf.add(Dense(7, activation='softmax'))

    cnn_clf.summary()

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
    print(classification_report(
        np.argmax(img_test_label, axis=1),
        np.argmax(test_pred, axis=1),
        target_names=['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
        )
    )

    # # Plot a gallery of 20 sample results
    # for i in range(20):
    #     plt.imshow(img_test[i], cmap='gray')
    #     plt.title("Predicted: {}\nTrue: {}".format(test_pred[i], img_test_label[i]))
    #     plt.show()
