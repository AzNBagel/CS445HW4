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


def gaussian_naive_bayes(filename="spambase.data"):
    raw_data = np.loadtxt(filename, delimiter=',')

    # Split raw data into examples.
    negatives = raw_data[raw_data[:, -1] == 0]
    positives = raw_data[raw_data[:, -1] == 1]

    neg_train = negatives[:(len(negatives) // 2)]
    neg_test = negatives[(len(negatives) // 2):]
    pos_train = positives[:(len(positives) // 2)]
    pos_test = positives[(len(positives) // 2):]

    pos_std = np.std(pos_train[:, :-1], axis=0, ddof=1)
    pos_mean = np.mean(pos_train[:, :-1], axis=0)
    neg_std = np.std(neg_train[:, :-1], axis=0, ddof=1)
    neg_mean = np.mean(neg_train[:, :-1], axis=0)


    neg_prior = len(pos_train)/len(neg_train)
    pos_prior = 1.0 - neg_prior
    print(neg_prior)
    print(pos_prior)

    pos_std[pos_std == 0] = .00000000000000001
    neg_std[neg_std == 0] = .00000000000000001

    test_data = np.vstack((pos_test, neg_test))

    # ###
    # Need to computer prior probabilities
    # ###
    pos_test = np.copy(test_data[:, :-1])

    pos_test = np.exp(-((pos_test - pos_mean) ** 2.0) / (2.0 * (pos_std ** 2)))
    pos_test[pos_test == 0] = .0000000000000000000000000000000000000001
    pos_test = (1.0 / (np.sqrt(2.0 * np.pi)) * pos_std) * pos_test

    pos_results = np.sum(np.log10(pos_test), axis=1)

    neg_test = np.copy(test_data[:, :-1])
    neg_test = np.exp(-(((neg_test - neg_mean) ** 2.0) / (2.0 * (neg_std ** 2))))
    neg_test[neg_test == 0] = .0000000000000000000000000000000000000001
    neg_test = (1.0 / (np.sqrt(2.0 * np.pi) * neg_std)) * neg_test

    neg_results = np.sum(np.log10(neg_test), axis=1)

    print(len(pos_test))
    print(len(neg_test))

    classification = []
    for i in range(len(pos_results)):
        if (np.log10(neg_prior) + neg_results[i]) > (np.log10(pos_prior) + pos_results[i]):
            classification.append(0)
        else:
            classification.append(1)

    print(classification)

    test_accuracy = metrics.accuracy_score(test_data[:, -1], classification)
    test_recall = metrics.recall_score(test_data[:, -1], classification)
    test_precision = metrics.precision_score(test_data[:, -1], classification)

    print("Accuracy: " + str(test_accuracy))
    print("Recall: " + str(test_recall))
    print("Precision: " + str(test_precision))

    return

gaussian_naive_bayes()
