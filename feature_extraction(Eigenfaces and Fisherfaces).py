'''
 Module for using fisherfaces to extract features:
    1. PCA (Principal component analysis)
    2. LDA (Linear Discriminant analysis)
    3. PCA(Eigenfaces) + LDA = Fisherfaces
'''


import math
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import preprocess_dataset as pd

from cv2 import cv2
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report



def constructRowMatrix(img_list):
    """
    A helper function for constructing row matrix given the input 
    data list of images. That is we want to transform the input image matrix
    into shape(n_samples, n_features)
    
    Args:
        img_list:    an input list of images in grayscale.
    Returns:
        row_matrix:  the desired row matrix into shape(n_samples, n_features).
    """
    # Check if the input image list is empty or not, if it is empty
    # print error and return
    if len(img_list) == 0:
        print("\nError: the input parameter [img_list] is empty\n")
        return
    
    # The img_list has been checked, we now construct row matrix from it
    row_matrix = np.asmatrix(img_list[0].flatten())
    for i in range(1, img_list.shape[0]):
        row_matrix = np.vstack([row_matrix, img_list[i].flatten()])

    # Return the result row_matrix
    return row_matrix


def plot_pca(pca_train, img_label_list):
    """
    A function to visualize the projected result using Principal component analysis (PCA)
    
    Args:
        pca_train:       the already transformed train data images.
        img_label_list:  an input list of corresponding image labels.
    Returns:
        None. (But generate visualization plot)
    """  
    ax = plt.subplot(111)

    # Plot each data point as a scatter on the plot
    for label, marker, color in zip(
        range(7), ('^', 's', 'o', 'h', 'v', '+', 'x'), ('red', 'blue', 'green', 'yellow', \
                                                                        'purple', 'orange', 'gray')):

        plt.scatter(x=pca_train[:, 0].real[img_label_list == label],
                y=pca_train[:, 1].real[img_label_list == label],
                marker=marker,
                color=color,
                alpha=0.5,
                label=(le.inverse_transform([label]))[0]
                )

    # set the two label titles
    plt.xlabel('PC1')
    plt.ylabel('PC2')

    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title('PCA: eigenfaces projection onto the first 2 principal components')

    # Hide axis ticks
    plt.tick_params(axis="both", which="both", bottom="off", top="off",  
            labelbottom="on", left="off", right="off", labelleft="on")

    # Remove axis spines
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)    

    plt.grid()
    plt.tight_layout
    plt.show()


def plot_step_lda(fisherfaces_train, img_label_list):
    """
    A function to visualize the projected result using Linear Discriminant analysis (LDA)
    
    Args:
        fisherfaces_train: the already transformed train data images.
        img_label_list:    an input list of corresponding image labels.
    Returns:
        None. (But generate visualization plot)
    """  
    ax = plt.subplot(111)


    for label, marker, color in zip(
        range(7), ('^', 's', 'o', 'h', 'v', '+', 'x'), ('red', 'blue', 'green', 'yellow', \
                                                                        'purple', 'orange', 'gray')):
        # Plot each data point as a scatter on the plot
        plt.scatter(x=fisherfaces_train[:, 0].real[img_label_list == label],
                y=fisherfaces_train[:, 1].real[img_label_list == label],
                marker=marker,
                color=color,
                alpha=0.5,
                label=(le.inverse_transform([label]))[0]
                )

    # set the two label titles
    plt.xlabel('LD1')
    plt.ylabel('LD2')

    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title('LDA: Fisherfaces projection onto the first 2 linear discriminants')

    # Hide axis ticks
    plt.tick_params(axis="both", which="both", bottom="off", top="off",  
            labelbottom="on", left="off", right="off", labelleft="on")

    # Remove axis spines
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)    

    plt.grid()
    plt.tight_layout
    plt.show()


def principalComponentAnalysis(train, test, img_label_list, num_components=0, show_original=False, \
                                                            show_projected=False, show_eigenfaces=False):
    """
    A function to extract a feature vector using Principal component analysis (PCA)
    
    Args:
        train:           an input training set for list of images in grayscale.
        test:            an input test set for list of images in grayscale.
        img_label_list:  an input list of corresponding image labels.
        num_components:  the number of components to keep, by default the value is 0;
                         Notes: 0 < num_components <= min(n_samples, n_features).
        show_original:   the input parameter to indicate whether we want to plot the original images list.
        show_projected:  the input parameter to indicate whether we want to plot the projected analysis result.
        show_eigenfaces: the input parameter to indicate whether we want to plot the eigenfaces images.
    Returns:
        pca_train:       a feature vector for the given input training img_list using PCA.
        pca_test:        a feature vector for the given input test img_list using PCA.
        pca:             the already trained pca instance.
    """  
    # If our input parameter indicates that we want to plot the original input images
    if show_original:
        i = 0
        eigen_plot = train[:20]
        for img in eigen_plot:
            i += 1
            plt.subplot(4, 5, i)
            plt.axis('off')
            plt.imshow(img, cmap='gray')
        plt.show()

    # Get num_elements, h and w of each signle image
    num_elements, h, w = train.shape
    # Deal with the situation that we don't have an input specified num_components
    if num_components == 0:
        num_lables = len(set(img_label_list))
        # Set num_components for PCA
        num_components = num_elements - num_lables

    # Initialize pca with parameter num_components
    pca = PCA(n_components=num_components)
    # Convert input image list into column matrix
    train_column_matrix = constructRowMatrix(train)
    test_column_matrix = constructRowMatrix(test)
    # Fit and transform the image list into PCA
    pca = PCA(n_components=num_components, svd_solver='randomized', whiten=True).fit(train_column_matrix)
    # Perform the dimension reduction on the test set
    pca_train = pca.transform(train_column_matrix)
    pca_test = pca.transform(test_column_matrix)
    # If our input parameter indicates that we want to plot the eigenfaces
    if show_eigenfaces:
        eigenfaces_show = pca.components_.reshape((num_components, h, w))
        i = 0
        eigen_plot = eigenfaces_show[:20]
        for img in eigen_plot:
            i += 1
            plt.subplot(4, 5, i)
            plt.axis('off')
            plt.imshow(img, cmap='gray')
        plt.show()

    # Plot the projected result onto the first two linear discriminants
    if show_projected: plot_pca(pca_train, img_label_list)

    # Return the result eigenfaces feature vector for both train and test sets
    return pca_train, pca_test, pca


def linearDiscriminantAnalysis(pca_train, pca_test, pca, img_label_list, class_components, \
                                    show_fisherfaces=False, show_projected=False, show_eigenfaces=False):
    """
    A function to extract a feature vector using Principal component analysis (PCA)
    
    Args:
        pca_train:         an input training set for list of images in grayscale.
        pca_test:          an input test set for list of images in grayscale.
        pca:               the already trained pca instance.
        img_label_list:    an input list of corresponding image labels.
        class_components:  the number of components to keep, by default the value is 0.   
        show_original:     the input parameter to indicate whether we want to plot the original images list.
        show_fisherfaces:  the input parameter to indicate whether we want to plot the fisherfaces.
        show_projected:    the input parameter to indicate whether we want to plot the projected analysis result.
        show_eigenfaces:   the input parameter to indicate whether we want to plot the eigenfaces images.
    Returns:
        lda_features:    a feature vector extracted from given input img_list using PCA.
    """  
    # Fit LDA algorithm and generate an instance of it
    lda = LinearDiscriminantAnalysis().fit(pca_train, img_label_list)
    lda.fit(pca_train, img_label_list)

    # Get the corresponding fisherfaces feature vectors
    fisherfaces_train = lda.transform(pca_train)
    fisherfaces_test = lda.transform(pca_test)

    # Plot the projected result onto the first two linear discriminants
    if show_projected: plot_step_lda(fisherfaces_train, img_label_list)

    # Plot the fisherfaces images
    if show_fisherfaces:
        for i in range(class_components - 1):
            fisherface_feature = pca.inverse_transform(lda.scalings_[:, i])
            fisherface_feature.shape = [48, 48]

            # Add 1 to i to satisfy the axis requirement in plt
            plt.subplot(1, 6, (i + 1))
            plt.axis('off')
            plt.imshow(fisherface_feature, cmap='jet')

        plt.show()
        plt.close()

    return fisherfaces_train, fisherfaces_test   


def fisherfaces(train, test, img_label_list):
    """
    A function to extract a feature vector using Principal component analysis (PCA)
    
    Args:
        train:              an input training set for list of images in grayscale.
        test:               an input test set for list of images in grayscale.
        img_label_list:     an input list of corresponding image labels.
    Returns:
        fisherfaces_train:  fisherfaces feature vector for the train set.
        fisherfaces_test:  fisherfaces feature vector for the test set.
    
    Note: fisherfaces_train and fisherfaces_test are the result fisherfaces vector by using the algorithm
          to do the feature extraction for each image. We will further use these two vectors to train the 
          CNN model and predict on the transformed test dataset.
    """  
    # We chain the PCA and LDA to get the result of fisherfaces
    # First apply PCA algorithm on the datasets: train and test
    pca_train, pca_test, pca = principalComponentAnalysis(img_train, img_test, img_train_label, \
                        num_components=625, show_original=True, show_projected=True, show_eigenfaces=True)

    # Use the result from PCA to get the fisherfaces result vectors for train and test datasets
    fisherfaces_train, fisherfaces_test = linearDiscriminantAnalysis(pca_train, pca_test, pca, img_train_label, \
        7, show_fisherfaces=True, show_projected=True, show_eigenfaces=True)

    # Return the fisherfaces_train and fisherfaces_test vectors
    return fisherfaces_train, fisherfaces_test



# Main program for testing the correctness of the implemented algorithm by using a simple one layer
# Muli-layer Perceptrons Classifier(MLP)
if __name__ == '__main__':
    # Uncomment below to see the test result
    pass
    # Load the datasets: train and test (also encoded labels)
    dataset_tuple_list = pd.load_dataset('CK+48')
    img_train, img_train_label, img_validation, img_validation_label, img_test, img_test_label, le = \
        pd.split_data(dataset_tuple_list)

    # Eigenfaces: Get the pca_train and pca_test feature vectors for further training and predicting
    pca_train, pca_test = principalComponentAnalysis(img_train, img_test, \
        img_train_label, num_components=625)[:2]

    # Fisherfaces: Get the fisherfaces_train and fisherfaces_test feature vectors for further training and predicting
    fisher_train, fisher_test = fisherfaces(img_train, img_test, img_train_label)

    # Train a nsimple neural network to test the correctness on the implemented algorithm
    print("\nFitting the classifier to the training set\n")
    clf = MLPClassifier(hidden_layer_sizes=(1024,), batch_size=256, verbose=True, early_stopping=True).fit(fisher_train, img_train_label)
    y_pred = clf.predict(fisher_test)
    print(classification_report(img_test_label, y_pred, target_names=['/anger', '/contempt', '/disgust', '/fear', '/happy', '/sadness', '/surprise']))