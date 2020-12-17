"""
 Module for using fisherfaces to extract features:
    1. PCA
    2. LDA
"""


import numpy as np
import data_preprocess as ppd

from cv2 import cv2
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA


def constructColumnMatrix(img_list):
    """
    A helper function for constructing column matrix given the input 
    data list of images.
    
    Args:
        img_list:       an input list of images in grayscale.
    Returns:
        column_matrix:  the desired column matrix.
    """
    # Check if the input image list is empty or not, if it is empty
    # print error and return
    if len(img_list) == 0:
        print("\nError: the input parameter [img_list] is empty\n")
        return
    
    # The img_list has been checked, we now construct column matrix from it
    h, w = img_list[0].shape[:2]
    column_matrix = np.empty([h * w, 0], dtype=img_list[0].dtype)
    for img in img_list:
        # Reshape each image to column vector and hstack each of them into 
        # column_matrix
        column_matrix = np.hstack([column_matrix, img.reshape(-1, 1)])

    # Return the result column_matrix
    return np.asmatrix(column_matrix)


# def PCA(img_list, num_components):
#     """
#     A function to extract a feature vector using Principal component analysis (PCA)
    
#     Args:
#         img_list:        an input list of images in grayscale.
#         num_components:  the number of components to keep.
#     Returns:
#         feature_vector:  a feature vector extracted from given input img_list using PCA.
#         eigenvectors:    the computed Wpca matrix.
#         eigenvalues:     the corresponding eigenvalue associated with each eigenvector in eigenvectors
#     """
#     # Convert the input img_list into a column matrix
#     img_column_matrix = constructColumnMatrix(img_list)

#     # Center the input image dataset 
#     # Make the computed mean vector as a column vector
#     mean_vector = np.mean(img_column_matrix, axis=1).reshape(-1, 1)
#     img_column_matrix = img_column_matrix -  mean_vector

#     # Perform Singular Value Decomposition(SVD) on img_column_matrix
#     eigenvectors, eigenvalues, variances = np.linalg.svd(img_column_matrix, full_matrices=False)

#     # Sort eigenvectors by eigenvalues in descending order
#     # (Negate the array, then the argsort will sort it in descending order)
#     # Reverse the array for descending order
#     descending_idx = np.argsort(eigenvalues)[::-1]
#     eigenvalues, eigenvectors = eigenvalues[descending_idx], eigenvectors[:, descending_idx]

#     # Pick num_components “best” new axes
#     projected_axes = eigenvectors[:, 0:num_components]
#     eigenvalues = eigenvalues[0: num_components]
#     # Finally convert singular values into eigenvalues 
#     eigenvalues = np.power(eigenvalues, 2) / img_column_matrix.shape[1]
#     # Compute the feature_vector by looping over the given input image data list
#     # Initialize the feature_vector
#     feature_vector = []
#     for img in img_list:
#         # The num_components of the observed vector x are then given by:
#         # y = W^{T}(x − μ)
#         # Reshape to a column vector
#         img_column_vector = img.reshape(-1, 1)
#         # Center each img
#         img_column_vector_centered = img_column_vector - mean_vector
#         img_projected = np.dot(np.transpose(projected_axes), img_column_vector_centered)
#         feature_vector.append(img_projected)

#     # Return the result feature_vector computed using PCA
#     return feature_vector, projected_axes, eigenvalues


def PCA(img_list, num_components):
    """
    A function to extract a feature vector using Principal component analysis (PCA)
    
    Args:
        img_list:        an input list of images in grayscale.
        num_components:  the number of components to keep.
    Returns:
        feature_vector:  a feature vector extracted from given input img_list using PCA.
        eigenvectors:    the computed Wpca matrix.
        eigenvalues:     the corresponding eigenvalue associated with each eigenvector in eigenvectors
    """  
    # Transform the datalist, flatten each image matrix within the img_list
    img_list_flatten = np.asmatrix(img_list[0].flatten())
    for i in range(1, img_list.shape[0]):
        img_list_flatten = np.vstack([img_list_flatten, img_list[i].flatten()])

    # Center the img data matrix with axis=0
    X_centered = img_list_flatten - np.mean(img_list_flatten, axis=0)

    # Compute the Covariance matrix from the img data matrix
    cov_Mat = np.cov(X_centered , rowvar=False)
     
    # Compute the eigenvalues and corresponding eigenvectors
    pca_eigenvals, pca_eigenvecs = np.linalg.eigh(cov_Mat)
     
    # Sort the eigenvalues in descending order
    sorted_idx = np.argsort(pca_eigenvals)[::-1]
    pca_eigenvals = pca_eigenvals[sorted_idx]
    pca_eigenvecs = pca_eigenvecs[:, sorted_idx]
     
    # Use given input num_components to get a subset of eigenvectors
    # that are associated with the num_components largest eigenvecotrs
    eigenvector_subset = pca_eigenvecs[:, 0:num_components]
     
    # Transform the img list data onto new axes
    X_reduced = np.dot(eigenvector_subset.T, X_centered.T)
     
    return X_reduced, eigenvector_subset


def LDA(img_list, img_labels):
    """
    A function to extract a feature vector using Principal component analysis (PCA)
    
    Args:
        img_list:        an input list of images in grayscale.
        img_labels:      an input list of encoded labels for the corresponding emotion image.
    Returns:
        feature_vector:  a feature vector extracted from given input img_list using LDA.
        eigenvectors:    the computed Wlda matrix.
        eigenvalues:     the corresponding eigenvalue associated with each eigenvector in eigenvectors
    """
    # Convert list into numpy array
    img_labels = np.array(img_labels)

    # Get the dimension of data point and the number of different labels
    d = img_list.shape[0]
    num_lables = len(set(img_labels))

    # Compute the mean vector and reshape into a column vector
    # and calculate the corresponding Sw, Sb (within, between) scatter matrices
    mean_vector = np.mean(img_list, axis=1).reshape(-1, 1)

    # Initialize Sw and Sb matrices
    Sw = np.zeros((d, d), dtype=np.float32)
    Sb = np.zeros((d, d), dtype=np.float32)
    for i in range(num_lables):
        Xi = img_list[:, np.where(img_labels == i)[0]]
        label_mean_vector = np.mean(Xi, axis=1).reshape(-1, 1)
        Sw = Sw + np.matmul((Xi - label_mean_vector), (Xi - label_mean_vector).T)
        Sb = Sb + Xi.shape[1] * np.matmul((label_mean_vector - mean_vector), (label_mean_vector - \
            mean_vector).T)

    # Compute the eigenvalues and eigenvectors for a general matrix
    general_matrix = np.matmul(np.linalg.inv(Sw), Sb)
    eigenvalues, eigenvectors = np.linalg.eig(general_matrix)
    # Sort eigenvectors by eigenvalues in descending order
    # (Negate the array, then the argsort will sort it in descending order)
    # Use [::-1] to reverse the sorting result and get the index of descending order
    descending_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[descending_idx], eigenvectors[:, descending_idx]

    # Only get the [num_labels - 1] non-zero eigenvalues
    eigenvalues = np.array(eigenvalues[: 2100 - 1].real, dtype=np.float32, copy=True)
    eigenvectors = np.matrix(eigenvectors[:, :2100 - 1].real, dtype=np.float32, copy=True)
    # Transform the img list data onto new axes
    X_lda = np.dot(eigenvectors.T, img_list)

    # Return the result feature_vector computed using LDA
    return X_lda, eigenvectors, eigenvalues


def Fisherfaces(img_list, img_labels):
    """
    A function to extract a feature vector using Principal component analysis (PCA)
    
    Args:
        img_list:        an input list of images in grayscale.
        img_labels:      an input list of encoded labels for the corresponding emotion image.
    Returns:
        feature_vector:  a feature vector extracted from given input img_list combined with
                         PCA and LDA.
    """
    # Convert list into numpy array
    img_labels = np.array(img_labels)

    # Get num_elements and num_lables from img_lables
    num_elements = img_labels.size
    num_lables = len(set(img_labels))

    # Get feature vectors by chaining PCA and LDA separately
    # Set num_components for PCA
    num_components_pca = num_elements - num_lables
    pca_feature, pca_eigenvec = PCA(img_list, 2207)
    lda_eigenvec = LDA(pca_feature, img_labels)[1]

    # pca_feature, pca_eigenvec, pca_eigenvals = PCA(img_list, img_labels, num_components_pca)
    # lda_feature, lda_eigenvec, lda_eigenvals = LDA(pca_feature, img_labels, num_components)
    # fisher_eigenvals = lda_eigenvals

    # Compute the new eigenspace as Wpca*Wlda
    fisher_eigenvec = pca_eigenvec * lda_eigenvec

    # Now compute the fisherfaces features
    fisher_feature_vector = []
    for img in img_list:
        # Reshape each img into a column vector
        img = img.reshape(-1, 1)
        img_projected = np.matmul(fisher_eigenvec.T, img)
        fisher_feature_vector.append(img_projected)

    # Return the result fisherfaces feature vector
    return fisher_feature_vector


if __name__ == '__main__':
    dataset_tuple_list = ppd.load_dataset('CK+48')
    img_train, img_train_label, img_validation, img_validation_label, img_test, img_test_label = \
        ppd.split_data(dataset_tuple_list)

    # Test accuracy for Fisherfaces

    # nsamples, nx, ny = img_train.shape
    # d2_train_dataset = img_train.reshape((nsamples, nx*ny))

    # pca = PCA(n_components=100, whiten=True).fit(d2_train_dataset)
    # X_train_pca = pca.transform(d2_train_dataset)
 
    # # apply PCA transformation
    # nsamples, nx, ny = img_test.shape
    # d2_train_dataset = img_test.reshape((nsamples, nx*ny))
    # X_test_pca = pca.transform(d2_train_dataset)

    feature_train = np.array(Fisherfaces(img_train, img_train_label))
    nsamples, nx, ny = feature_train.shape
    feature_train = feature_train.reshape((nsamples, nx*ny))

    feature_test = np.array(Fisherfaces(img_test, img_test_label))
    nsamples, nx, ny = feature_test.shape
    feature_test = feature_test.reshape((nsamples, nx*ny))

    # Train a neural network
    print("\nFitting the classifier to the training set\n")
    clf = MLPClassifier(hidden_layer_sizes=(1024,), batch_size=256, verbose=True, early_stopping=True).fit(feature_train, img_train_label)
    y_pred = clf.predict(feature_test)
    print(classification_report(img_test_label, y_pred, target_names=['/anger', '/contempt', '/disgust', '/fear', '/happy', '/sadness', '/surprise']))
