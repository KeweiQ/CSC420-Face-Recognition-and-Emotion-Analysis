'''
 The module to train each model for the
 facial expression analysis. Models including:
    - Convolutional Neural Network (CNN)
    -
'''


import keras
import construct_model

from keras.utils import to_categorical

def train_model(model, model_kind, train, train_label, validation, validation_label, algorithm):
    """
    The function for training the cnn model with the input train and
    validation datasets

    Args:
        model:              the input model for the purpose of training.
        model_kind:         kind of the input model.
        train:              the train dataset.
        train_label:        the labels for the train dataset.
        validation:         the validation dataset.
        validation_label:   the labels for the validation dataset.
        algorithm:          the feature algorithm that we want to perform.

    Returns:
        model_trained:      the instance of the model that is already trained.
    """
    if model_kind == 'cnn':
        if algorithm == 'eigenfaces':
            # Reshape the input datasets to accord the cnn model
            train_adjust = train.reshape(-1, 1, 625)
            validation_adjust = validation.reshape(-1, 1, 625)

            # One-hot encoding for the labels
            train_label_hotcoder = to_categorical(train_label)
            validation_label_hotcoder = to_categorical(validation_label)

            # Train the neural network model
            model.fit(train_adjust, train_label_hotcoder,
                    batch_size=256,
                    epochs=100,
                    verbose=1,
                    validation_data=(validation_adjust, validation_label_hotcoder)
                    )

        elif algorithm == 'fisherfaces':
            # Reshape the input datasets to accord the cnn model
            train = train.reshape(-1, 1, 6)
            validation = validation.reshape(-1, 1, 6)

            # One-hot encoding for the labels
            train_label_hotcoder = to_categorical(train_label)
            validation_label_hotcoder = to_categorical(validation_label)

            # Train the neural network model
            model.fit(train, train_label_hotcoder,
                        batch_size=256,
                        epochs=100,
                        verbose=1,
                        validation_data=(validation, validation_label_hotcoder)
                        )

        # Return the instance of the trained model
        return model

    elif model_kind == 'svm':
        pass
    elif model_kind == 'adaboost':
        pass
