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

    neg_train = negatives[:(len(negatives)//2)]
    neg_test = negatives[(len(negatives)//2):]
    pos_train = positives[:(len(positives)//2)]
    pos_test = positives[(len(positives)//2):]

    pos_std = np.std(pos_train[:, :-1], axis=0, ddof=1)
    pos_mean = np.mean(pos_train[:, :-1], axis = 0)
    neg_std = np.std(neg_train[:, :-1], axis=0, ddof=1)
    neg_mean = np.mean(neg_train[:, :-1], axis = 0)

    pos_std[pos_std == 0.0] = .000000000001
    neg_std[neg_std == 0.0] = .000000000001

    test_data = np.vstack((pos_test, neg_test))

    print(len(test_data))
    print(test_data)



    # Need to iterate over columns based on the final
    # So what do we need to do here for each P val.
    # For each row we could apply the function
    pos_test = np.copy(test_data[:,:-1])


    pos_test = np.exp(-((pos_test - pos_mean) ** 2.0)/(2.0 * pos_std ** 2))
    pos_test[pos_test == 0.0] = .000000000001
    pos_test = (1.0 / ((np.sqrt(2.0 * np.pi))) * pos_std) * pos_test

    pos_results = np.sum(np.log10(pos_test), axis=1)
    print(pos_test)


    print(pos_results)


    neg_test = np.copy(test_data[:,:-1])
    neg_test = np.exp(-((neg_test - neg_mean) ** 2.0)/(2.0 * neg_std ** 2))
    neg_test[neg_test == 0.0] = .000000000001
    neg_test = (1.0 / ((np.sqrt(2.0 * np.pi))) * neg_std) * neg_test

    neg_results = np.sum(np.log10(neg_test), axis=1)






    return




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

    std_array = np.std(training_data[:, :-1], axis=0, ddof=1)
    mean_array = np.mean(training_data[:, :-1], axis = 0)

    scalar = preprocessing.StandardScaler().fit(training_data[:, :-1])

    training_data[:, :-1] = scalar.transform(training_data[:, :-1])
    test_data[:, :-1] = scalar.transform(test_data[:, :-1])

    mean = scalar.mean_

    std = scalar.scale_


    print("*********Standard Scaler MEAN:")
    print(mean)
    print(len(mean))

    print("*********NP Mean: ")
    print(mean_array)
    print(len(mean_array))

    print("*********Standard Scaler STD:")
    print(std)
    print(len(std))
    print("*********NP Std ddof=1:")
    print(std_array)
    print(len(std_array))



    return training_data, test_data, mean, std

gaussian_naive_bayes()