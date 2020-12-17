'''
 Module to evaluate different machine learning models:
    1.
    2.
    3.
'''


import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import KFold
from itertools import cycle
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, roc_curve, auc
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


class KNearestNeighbor(object):
    '''
    K Nearest Neighbor classifier
    '''

    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_norm = (self.train_data ** 2).sum(axis=1).reshape(-1, 1)
        self.train_labels = train_labels

    def l2_distance(self, test_point):
        '''
        Compute L2 distance between test point and each training point

        Input: test_point is a 1d numpy array
        Output: dist is a numpy array containing the distances between the test point and each training point
        '''
        # Process test point shape
        test_point = np.squeeze(test_point)
        if test_point.ndim == 1:
            test_point = test_point.reshape(1, -1)
        assert test_point.shape[1] == self.train_data.shape[1]

        # Compute squared distance
        test_norm = (test_point ** 2).sum(axis=1).reshape(1, -1)
        dist = self.train_norm + test_norm - 2 * self.train_data.dot(test_point.transpose())
        return np.squeeze(dist)

    def query_knn(self, test_point, k):
        '''
        Query a single test point using the k-NN algorithm

        You should return the digit label provided by the algorithm
        '''

        # calculate L2 distances between test_point and all train points
        distances = self.l2_distance(test_point)
        # get the indices of k smallest distances
        ksmt_index = np.argsort(distances)[:k]

        # get labels of corresponding k smallest train points
        labels = np.zeros(k)
        for i in range(0, k):
            labels[i] = self.train_labels[ksmt_index[i]]

        # find the most common label in label list
        c = Counter(labels)
        digit = max(labels, key=c.get)

        return digit

    def predict_proba(self, test_point, k):
        # calculate L2 distances between test_point and all train points
        distances = self.l2_distance(test_point)
        # get the indices of k smallest distances
        ksmt_index = np.argsort(distances)[:k]

        # get labels of corresponding k smallest train points
        labels = np.zeros(k)
        for i in range(0, k):
            labels[i] = self.train_labels[ksmt_index[i]]

        # find the number of each label in k neighbours
        c = Counter(labels)
        count = np.zeros(10)
        for i in range(10):
            count[i] = c[i]

        # find the frequency of each label in k neighbours
        freq = np.zeros(10)
        for i in range(10):
            freq[i] = count[i] / 10

        return freq


def cross_validation(train_data, train_labels, k_range=np.arange(1, 16)):
    '''
    Perform 10-fold cross validation to find the best value for k

    Note: Previously this function took knn as an argument instead of train_data,train_labels.
    The intention was for students to take the training data from the knn object - this should be clearer
    from the new function signature.
    '''
    avg_accuracies = np.zeros(15)
    fold_accuracies = []

    for k in k_range:
        # perform 10 fold cross validation
        kf = KFold(n_splits=10)
        accuracies = np.zeros(10)
        i = 0
        for train, valid in kf.split(train_labels):
            # split data
            x_train, x_vaild, y_train, y_valid = train_data[train], train_labels[train], train_data[valid], \
                                                 train_labels[valid]
            # train model
            knn = KNearestNeighbor(x_train, x_vaild)
            # get predict accuracy
            accuracies[i] = classification_accuracy(knn, k, y_train, y_valid)
            i += 1

        avg_accuracies[k - 1] = np.mean(accuracies)
        fold_accuracies.append(np.copy(accuracies))

    # find best k and print accuracies
    k = int(np.argmax(avg_accuracies))
    print('Train classification accuracy for each fold: ', fold_accuracies[k])
    print('Average accuracy across folds: ', avg_accuracies[k])

    return k


def classification_accuracy(knn, k, eval_data, eval_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''
    predict_arr = np.zeros(len(eval_data))
    for i in range(0, len(eval_data)):
        predict_arr[i] = knn.query_knn(eval_data[i], k)

    accuracy = accuracy_score(predict_arr, eval_labels)
    return accuracy


def classification_frequency(knn, k, eval_data):
    '''
    Evaluate the classification frequency of knn on the given 'eval_data'
    using the labels
    '''
    freq_arr = []
    for i in range(0, len(eval_data)):
        freq_arr.append(knn.predict_proba(eval_data[i], k))

    return np.asarray(freq_arr)


def knn(train_data, train_labels, test_data, test_labels):
    knn = KNearestNeighbor(train_data, train_labels)

    # get predictions on test data and plot roc curve
    predict_arr = np.zeros(len(test_data))
    for i in range(0, len(test_data)):
        predict_arr[i] = knn.query_knn(test_data[i], 3)
    score = classification_frequency(knn, 3, test_data)
    plot_ROC(test_labels, score)

    # print other prediction metrics
    print('KNN MSE: ', mean_squared_error(test_labels, predict_arr))
    print('KNN confusion matrix:\n', confusion_matrix(test_labels, predict_arr))
    print('KNN classification report:\n', classification_report(test_labels, predict_arr))


def nerual_network(train_data, train_labels, test_data, test_labels):
    # hyperparameters for gird search
    param_grid = {
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'learning_rate': ['constant', 'invscaling', 'adaptive']
    }

    # grid search
    grid = GridSearchCV(MLPClassifier(), param_grid, verbose=2, n_jobs=1)
    # train model with best hyperparameters
    grid.fit(train_data, train_labels)
    print(grid.best_params_)

    # get predictions on test data and plot roc curve
    grid_predictions = grid.predict(test_data)
    predict_score = grid.predict_proba(test_data)
    print(predict_score)
    plot_ROC(test_labels, predict_score)

    # print other prediction metrics
    print('Nerual network test accuracy: ', accuracy_score(test_labels, grid_predictions))
    print('Nerual network MSE: ', mean_squared_error(test_labels, grid_predictions))
    print('Nerual network confusion matrix:\n', confusion_matrix(test_labels, grid_predictions))
    print('Nerual network classification report:\n', classification_report(test_labels, grid_predictions))


def SVM(train_data, train_labels, test_data, test_labels):
    # hyperparameters for gird search
    param_grid = {'C': [1, 10], 'gamma': [0.5, 0.1], 'kernel': ['rbf', 'poly', 'sigmoid']}

    # grid search
    grid = GridSearchCV(SVC(), param_grid, verbose=2, n_jobs=1)
    # train model with best hyperparameters
    grid.fit(train_data, train_labels)
    print(grid.best_params_)

    # get predictions on test data and plot roc curve
    grid_predictions = grid.predict(test_data)
    predict_score = grid.decision_function(test_data)
    plot_ROC(test_labels, predict_score)

    # print other prediction metrics
    print('SVM MSE:', mean_squared_error(test_labels, grid_predictions))
    print('SVM confusion matrix:\n', confusion_matrix(test_labels, grid_predictions))
    print('SVM classification report:\n', classification_report(test_labels, grid_predictions))


def ada_boost(train_data, train_labels, test_data, test_labels):
    # hyperparameters for gird search
    param_grid = {'n_estimators': [500, 1000], 'learning_rate': [0.01, 0.1]}

    # grid search
    grid = GridSearchCV(AdaBoostClassifier(), param_grid, verbose=2, n_jobs=1)
    # train model with best hyperparameters
    grid.fit(train_data, train_labels)
    print(grid.best_params_)

    # get predictions on test data and plot roc curve
    grid_predictions = grid.predict(test_data)
    predict_score = grid.decision_function(test_data)
    plot_ROC(test_labels, predict_score)

    # print other prediction metrics
    print('AdaBoost MSE: ', mean_squared_error(test_labels, grid_predictions))
    print('AdaBoost confusion matrix:\n', confusion_matrix(test_labels, grid_predictions))
    print('AdaBoost classification report:\n', classification_report(test_labels, grid_predictions))


def plot_ROC(test_labels, grid_predictions):
    # Compute ROC curve and ROC area for each class
    n_classes = 10
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        true = label_binarize(test_labels, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        fpr[i], tpr[i], _ = roc_curve(true[:, i], grid_predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    colors = cycle(['purple', 'orange', 'green', 'blue', 'pink', 'red', 'yellow', 'aqua', 'lightblue', 'lightgreen'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, label='Class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC curve')
    plt.legend(loc="lower right")
    plt.show()


def main():
    # knn(train_data, train_labels, test_data, test_labels)
    # nerual_network(train_data, train_labels, test_data, test_labels)
    # SVM(train_data, train_labels, test_data, test_labels)
    # ada_boost(train_data, train_labels, test_data, test_labels)
    pass


if __name__ == '__main__':
    main()
