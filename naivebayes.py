from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import metrics

"""
Andrew McCann
CS445 Machine Learning
Homework #4
Melanie Mitchell
"""

















def load_spam_data(filename="spambase.data"):
    """
    Generates data from text file into np arrays for analyses.

    :param filename: filename of data text file.
    :return: training_set, test data for training and testing our SVM.
    """
    raw_data = np.loadtxt(filename, delimiter=',')

    # Split raw data into examples.
    negatives = raw_data[raw_data[:, -1] == 0]
    positives = raw_data[raw_data[:, -1] == 1]

    # To preserve the ratio of + and - we split prior to shuffling
    negatives1 = negatives[:(len(negatives)//2)]
    negatives2 = negatives[(len(negatives)//2):]
    positives1 = positives[:(len(positives)//2)]
    positives2 = positives[(len(positives)//2):]

    print(len(negatives1))
    print(len(negatives2))
    print(len(positives1))
    print(len(positives2))

    test_data = np.vstack((positives2, negatives2))
    training_data = np.vstack((positives1, negatives1))

    np.random.shuffle(training_data)

    scalar = preprocessing.StandardScaler().fit(training_data[:, :-1])

    training_data[:, :-1] = scalar.transform(training_data[:, :-1])
    test_data[:, :-1] = scalar.transform(test_data[:, :-1])

    mean = scalar.mean_
    std = scalar.scale_

    print(mean)
    print(len(mean))
    print(std)
    print(len(std))

    return training_data, test_data, mean, std

load_spam_data()