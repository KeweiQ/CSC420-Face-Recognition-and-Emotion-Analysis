"""
 Module for building the face recognition and facial expression analysis system
"""


import data_preprocess as dp
import face_detection as fd
import feature_extraction as fe
import model_construction as mc
import model_evaluation as me
import model_construction_old as mco


def main():
    """
    The main program for building the system. And we support following kinds of model:
        1. Convolutional Neural Network (CNN)
        2. Support Vector Machine (SVM)
        3. Adaboost
        4. Multilayer Perceptron (MLP)

    Args:
        None.

    Returns:
        None.

    Note: we'll construct and evaluate several different models for emotion
          analysis. And also, we combine the Principal Component Analysis (PCA,
          eigenfaces) with the Linear Discriminant Analysis (LDA), and use the
          combination (fisherfaces) to do feature extraction for each image to
          improve our model performances.
    """
    # Give a prompt for user to specify their wanted model
    model_type = input("Please select a kind of model from the following: 'cnn', 'svm', 'adaboost', 'mlp':\n")
    while model_type not in ['cnn', 'svm', 'adaboost', 'mlp']:
        print("Your input is not correct, please try again:\n")
        model_type = input("Please select a kind of model from the following: 'cnn', 'svm', 'adaboost', 'mlp:\n")
    print(f"You entered {model_type}\n")

    # Give a prompt for user to specify their wanted dataset
    dataset = input("Please select a dataset that you want to perform from the following: 'CK+48', 'fer2013'.\n")
    while dataset not in ['CK+48', 'fer2013']:
        print("Your input is not correct, please try again:\n")
        dataset = input("Please select a dataset that you want to perform from the following: 'CK+48', 'fer2013'.\n")
    print(f"You entered {dataset}\n")

    # Give a prompt for user to specify their wanted feature extraction algorithm
    algorithm = input("Please select an feature extraction algorithm that you want to use from the following: 'eigenfaces', 'fisherfaces'.\n")
    while algorithm not in ['eigenfaces', 'fisherfaces']:
        print("Your input is not correct, please try again:\n")
        algorithm = input("Please select an feature extraction algorithm that you want to use from the following: 'eigenfaces', 'fisherfaces'.\n")
    print(f"You entered {algorithm}")

    # Load the dataset into a shuffled list of tuples
    dataset_tuple_list = dp.load_dataset(dataset)

    # # Test to see the loading result
    # for data_tuple in dataset_tuple_list:
    #     print(data_tuple)

    # Split the dataset into train, test, validation and their corresponding labels
    img_train, img_train_label, img_validation, img_validation_label, img_test, img_test_label, le = \
        dp.split_data(dataset_tuple_list)

    if model_type == 'cnn':
        # Do the feature extraction algorithm on the splitted datasets
        if algorithm == 'eigenfaces':
            # Eigenfaces: Get the pca_train and pca_test feature vectors for further training and predicting
            pca_train, pca_test, pca_validation = fe.principalComponentAnalysis(img_train, img_test, img_validation,
                                                                                img_train_label, le, num_components=625)[:3]

            # Construct and train the selected model with the input train and validation datasets
            model_trained = mc.train_model(model_type, pca_train, img_train_label, pca_validation, img_validation_label,
                                           algorithm)

            # Perform me on the trained model with the test dataset
            me.evaluate_model(model_trained, model_type, pca_test, img_test_label, algorithm)

        elif algorithm == 'fisherfaces':
            # Fisherfaces: Get the fisherfaces_train and fisherfaces_test feature vectors for further training and predicting
            fisher_train, fisher_test, fisher_validation = fe.fisherfaces(img_train, img_test, img_validation,
                                                                          img_train_label, le)

            # Construct and train the selected model with the input train and validation datasets
            model_trained = mc.train_model(model_type, fisher_train, img_train_label, fisher_validation, img_validation_label,
                                           algorithm)

            # Perform me on the trained model with the test dataset
            me.evaluate_model(model_trained, model_type, fisher_test, img_test_label, algorithm)

    elif model_type == 'svm':
        pass

    elif model_type == 'adaboost':
        pass

    elif model_type == 'mlp':
        pass


# Main program
if __name__ == '__main__':
    main()