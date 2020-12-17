"""
 Module for building different type of models for the facial expression analysis.
 Model types including:
    - Convolutional Neural Network (CNN)
    - Support Vector Machine (SVM)
    - AdaBoost
    - Multilayer Perceptron (MLP)
"""


from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
from keras.layers.advanced_activations import LeakyReLU


def train_model(model_type, train, train_label, validation=None, validation_label=None, algorithm=None):
    """
    The function for training the cnn model with the input train and
    validation datasets

    Args:
        model_type:         type of the input model.
        train:              the train dataset.
        train_label:        the labels for the train dataset.
        validation:         the validation dataset. (CNN only)
        validation_label:   the labels for the validation dataset. (CNN only)
        algorithm:          the feature algorithm that we want to perform. (CNN only)

    Returns:
        model_trained:      the instance of the model that is already trained.
    """
    if model_type == 'cnn':
        if algorithm == 'eigenfaces':
            # build cnn model
            model = construct_cnn(algorithm)

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
            # build cnn model
            model = construct_cnn(algorithm)

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

        else:
            print("ERROR: invalid algorithm for CNN!")
            return None

        # Return the instance of the trained model
        return model

    elif model_type == 'svm':
        # hyperparameters for gird search
        param_grid = {'C': [1, 10], 'gamma': [0.5, 0.1], 'kernel': ['rbf', 'poly', 'sigmoid']}

        # grid search
        grid = GridSearchCV(SVC(), param_grid, verbose=2, n_jobs=1)
        # train model with best hyperparameters
        grid.fit(train, train_label)
        print(grid.best_params_)

        return grid

    elif model_type == 'adaboost':
        # hyperparameters for gird search
        param_grid = {'n_estimators': [500, 1000], 'learning_rate': [0.01, 0.1]}

        # grid search
        grid = GridSearchCV(AdaBoostClassifier(), param_grid, verbose=2, n_jobs=1)
        # train model with best hyperparameters
        grid.fit(train, train_label)
        print(grid.best_params_)

        return grid

    elif model_type == 'mlp':
        # hyperparameters for gird search
        param_grid = {
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
            'learning_rate': ['constant', 'invscaling', 'adaptive']
        }

        # grid search
        grid = GridSearchCV(MLPClassifier(), param_grid, verbose=2, n_jobs=1)
        # train model with best hyperparameters
        grid.fit(train, train_label)
        print(grid.best_params_)

        return grid

    else:
        print("ERROR: invalid model type!")
        return None


def construct_cnn(algorithm):
    """
    A function for constructing a Convolutional Neural Network (CNN)
    model for emotion classification

    Args:
        model_kind:  the input kind of the model that we want to build.
        algorithm:   the feature algorithm that we want to perform.
    Returns:
        [model]:       the generated models aligned with the model_kind.
    """
    # Construct the convolutional neural network
    cnn_clf = Sequential()
    # The first layer
    if algorithm == 'eigenfaces':
        cnn_clf.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(1, 625), padding='same'))
    elif algorithm == 'fisherfaces':
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

    return cnn_clf
