'''
 Module for preprocessing the CK+48 dataset:
    1. Load dataset
    2. Shuffle dataset
    3. Split into training, validation and test datasets
'''


import glob
import numpy as np

from cv2 import cv2
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def load_dataset(dataset):
    """
    A function that loads all data images and corresponding labels from the CK+48 
    dataset folder

    Args:
        dataset: input dataset name(namely CK+48).

    Returns:
        dataset_tuple_list: a already shuffled list of tuples that each tuple containing a facial 
                            emotional image and a corresponding label.
    """
    # Initialize the result array to contain the loaded tuples
    dataset_tuple_list = []
    # Create label list for iteration
    emotion_labels = ['/anger', '/contempt', '/disgust', '/fear', '/happy', '/sadness', '/surprise']
    # Loop over each emotion labels to load all the images of the CK+48 dataset in tuple
    for label in emotion_labels:
        path = dataset + label + '/*.png'
        for image in glob.glob(path):
            img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            # Remove the '/' character from 'label' iterator
            dataset_tuple_list.append((img, label[1:]))

    # Shuffle all the loaded images for further splitting into train and test dataset
    np.random.shuffle(dataset_tuple_list)

    return dataset_tuple_list


def split_data(dataset_tuple_list):
    """
    A function that splits the entire dataset randomly into 70% training, 15% validation, 
    and 15% test examples.

    Args:
        train_test_split: a already shuffled list of tuples that each tuple containing a facial 
                          emotional image and a corresponding label generated from load_dataset().

    Returns:
        img_train:              the array stores the training images
        img_train_label:        the array stores the corresponding target for the training images
        img_validation:         the array stores the validation images
        img_validation_label:   the array stores the corresponding target for the validation images
        img_test:               the array stores the test images
        img_test_label:         the array stores the corresponding target for the test images
        le:                     a labelEncoder instance for further reverse transform back number 
                                labels into string.
    """
    # Get images and target labels from the input dataset_tuple_list
    images = []
    targets = []
    for data_tuple in dataset_tuple_list:
        images.append(data_tuple[0])
        targets.append(data_tuple[1])

    # Encode string labels into numbers
    le = preprocessing.LabelEncoder()
    encodes_targets = le.fit_transform(targets)

    # Split into train, validation and test data sets
    img_train, img_left, img_train_label, img_label_left = train_test_split(images, encodes_targets, 
                                        train_size=0.70, test_size=0.30, random_state=42)
    
    img_validation, img_test, img_validation_label, img_test_label = train_test_split(img_left, \
        img_label_left, train_size=0.50, test_size=0.50, random_state=42)

    # Convert them all into numpy array
    img_train = np.array(img_train)
    img_train_label = np.array(img_train_label)
    img_validation = np.array(img_validation)
    img_validation_label = np.array(img_validation_label)
    img_test = np.array(img_test)
    img_test_label = np.array(img_test_label)
    
    # Return the splitted datasets
    return img_train, img_train_label, img_validation, img_validation_label, img_test, img_test_label, le


# Main program
if __name__ == '__main__':
    # Load the dataset into a shuffled list of tuples
    dataset_tuple_list = load_dataset('CK+48')

    # # Test to see the loading result
    # for data_tuple in dataset_tuple_list:
    #     print(data_tuple)

    img_train, img_train_label, img_validation, img_validation_label, img_test, img_test_label, le = \
        split_data(dataset_tuple_list)
